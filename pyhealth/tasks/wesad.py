"""WESAD wrist EDA stress-classification task."""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from typing import Any, ClassVar
from zipfile import ZipFile

import numpy as np
import torch

from pyhealth.tasks import BaseTask

EDA_SAMPLING_RATE = 4
LABEL_SAMPLING_RATE = 700
LABEL_CODES = {
    0: "undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "transient_5",
    6: "transient_6",
    7: "transient_7",
}


class WESADStressClassification(BaseTask):
    """Classify full, non-overlapping wrist EDA windows as baseline or stress.

    The task reads each recording from the ``archive_path`` and ``member_path``
    attributes indexed by :class:`~pyhealth.datasets.WESADDataset`. WESAD wrist
    EDA is sampled at 4 Hz and its labels at 700 Hz. Labels remain at 700 Hz
    while windows are aligned: each EDA sample corresponds to 175 label
    samples, and both are cut using the same recording-relative boundaries.

    A window is emitted only when every source label in it is code 1 (baseline)
    or every source label is code 2 (stress). Mixed-label windows, labels for
    other states (0 and 3-7), incomplete trailing windows, and incomplete
    label-to-EDA intervals are discarded.

    Args:
        window_seconds: Positive window duration that corresponds to a whole
            number of 4 Hz EDA samples. Defaults to 60 seconds.

    Each sample contains a ``signal`` tensor of shape ``(1, window_samples)``,
    a string ``label`` (``"baseline"`` or ``"stress"``), and recording/window
    metadata. The signal is the raw wrist EDA in the units stored by WESAD.

    Examples:
        >>> from pyhealth.datasets import WESADDataset
        >>> from pyhealth.tasks import WESADStressClassification
        >>> dataset = WESADDataset(root="data/wesad")  # doctest: +SKIP
        >>> samples = dataset.set_task(
        ...     WESADStressClassification(window_seconds=60)
        ... )  # doctest: +SKIP
    """

    task_name: str = "WESAD_stress_classification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "multiclass"}
    label_codes: ClassVar[dict[int, str]] = LABEL_CODES

    def __init__(self, window_seconds: float = 60) -> None:
        super().__init__()
        if isinstance(window_seconds, bool):
            raise ValueError("window_seconds must be a positive number.")
        try:
            duration = float(window_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("window_seconds must be a positive number.") from exc
        exact_samples = duration * EDA_SAMPLING_RATE
        if (
            not math.isfinite(duration)
            or duration <= 0
            or not math.isclose(exact_samples, round(exact_samples), abs_tol=1e-8)
            or round(exact_samples) < 1
        ):
            raise ValueError(
                "window_seconds must be positive and align to a whole "
                f"number of {EDA_SAMPLING_RATE} Hz EDA samples."
            )
        self.window_seconds = duration
        self.window_samples = round(exact_samples)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Create labeled EDA windows for one WESAD participant."""
        samples: list[dict[str, Any]] = []
        label_samples_per_eda = LABEL_SAMPLING_RATE // EDA_SAMPLING_RATE
        if LABEL_SAMPLING_RATE % EDA_SAMPLING_RATE:
            raise ValueError("WESAD label and EDA sampling rates are not aligned.")

        for event in patient.get_events(event_type="eda"):
            archive_path = getattr(event, "archive_path", None)
            member_path = getattr(event, "member_path", None)
            if not archive_path or not member_path:
                raise ValueError(
                    f"WESAD event for {patient.patient_id} is missing "
                    "archive_path or member_path."
                )

            with ZipFile(archive_path) as archive:
                with archive.open(member_path) as recording_file:
                    # The original recordings contain Python 2 NumPy pickles.
                    recording = pickle.load(recording_file, encoding="latin1")
            eda, labels = self._extract_signals(recording, member_path)

            label_eda_samples = labels.size // label_samples_per_eda
            if abs(eda.size - label_eda_samples) > 1:
                raise ValueError(
                    f"WESAD recording {member_path} has mismatched EDA and label "
                    f"durations ({eda.size} EDA samples vs. {label_eda_samples} "
                    "label-aligned samples)."
                )
            aligned_eda_samples = min(eda.size, label_eda_samples)
            window_count = aligned_eda_samples // self.window_samples
            for window_index in range(window_count):
                start = window_index * self.window_samples
                end = start + self.window_samples
                window_labels = labels[
                    start * label_samples_per_eda : end * label_samples_per_eda
                ]
                if np.all(window_labels == 1):
                    label = "baseline"
                elif np.all(window_labels == 2):
                    label = "stress"
                else:
                    continue

                signal = np.array(eda[start:end], dtype=np.float32, copy=True)
                samples.append(
                    {
                        "patient_id": str(patient.patient_id),
                        "record_id": str(
                            getattr(event, "record_id", patient.patient_id)
                        ),
                        "window_index": window_index,
                        "start_seconds": start / EDA_SAMPLING_RATE,
                        "signal": torch.from_numpy(signal[None, :]),
                        "label": label,
                    }
                )
        return samples

    @staticmethod
    def _extract_signals(
        recording: Any, member_path: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and return one recording's wrist EDA and 700 Hz labels."""
        if not isinstance(recording, Mapping):
            raise ValueError(f"WESAD recording {member_path} is not a mapping.")
        wrist = recording.get("signal")
        if not isinstance(wrist, Mapping) or not isinstance(
            wrist.get("wrist"), Mapping
        ):
            raise ValueError(
                f"WESAD recording {member_path} is missing wrist signal data."
            )
        if "EDA" not in wrist["wrist"] or "label" not in recording:
            raise ValueError(
                f"WESAD recording {member_path} must contain wrist EDA and labels."
            )

        try:
            eda = np.asarray(wrist["wrist"]["EDA"], dtype=np.float32).squeeze()
            raw_labels = np.asarray(recording["label"]).squeeze()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"WESAD recording {member_path} has invalid EDA or labels."
            ) from exc
        if eda.ndim != 1 or raw_labels.ndim != 1:
            raise ValueError(
                f"WESAD recording {member_path} EDA and labels must be 1D."
            )
        if not np.issubdtype(raw_labels.dtype, np.number):
            raise ValueError(f"WESAD labels in {member_path} must be numeric.")
        if not np.all(np.isfinite(raw_labels)):
            raise ValueError(
                f"WESAD labels in {member_path} contain non-finite values."
            )
        labels = raw_labels.astype(np.int64)
        if not np.array_equal(raw_labels, labels):
            raise ValueError(f"WESAD labels in {member_path} must be integer codes.")
        unknown_codes = sorted(set(np.unique(labels).tolist()) - set(LABEL_CODES))
        if unknown_codes:
            raise ValueError(
                f"WESAD labels in {member_path} contain unsupported codes: "
                f"{unknown_codes}."
            )
        if not np.all(np.isfinite(eda)):
            raise ValueError(f"WESAD EDA in {member_path} contains non-finite values.")
        return eda, labels
