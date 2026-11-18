"""WESAD recording index for EDA stress-classification tasks.

Author: William D'Onofrio
Paper: Contrastive Learning of Electrodermal Activity Representations for
    Stress Detection. https://proceedings.mlr.press/v209/matton23a.html
Description: Index WESAD recordings for downstream signal tasks.
"""

import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .base_dataset import BaseDataset
from .configs import load_yaml_config
from .metadata_csv import write_metadata_csv

_CSV_NAME = "wesad-pyhealth.csv"
_ATTRIBUTES = ["record_id", "archive_path", "member_path"]
_RECORDING = re.compile(r"WESAD/(?P<subject>S[1-9][0-9]*)/(?P=subject)\.pkl")


class WESADDataset(BaseDataset):
    """Index one ``eda`` event per participant in ``WESAD.zip``.

    Records contain archive/member paths; signals are not loaded. Initialization
    rebuilds the CSV index and invalidates processed caches, even if unchanged.
    Metadata falls back to PyHealth's home cache when the root is not writable.

    Args:
        root: Directory containing ``WESAD.zip``.
        dataset_name: Dataset name. Defaults to ``"wesad"``.
        config_path: YAML config retaining the default ``eda`` table mapping.
        cache_dir: Root directory for processed-data caches.

    Examples:
        >>> from pyhealth.datasets import WESADDataset
        >>> dataset = WESADDataset(root="data/wesad")  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "wesad.yaml")
        config = load_yaml_config(config_path)
        table = config.tables.get("eda")
        if (
            table is None
            or table.file_path != _CSV_NAME
            or table.patient_id != "subject_id"
            or table.timestamp is not None
            or table.attributes != _ATTRIBUTES
            or table.join
        ):
            raise ValueError("The eda table must retain the WESAD metadata mapping.")

        self.raw_root = Path(root).expanduser().resolve()
        metadata_path = self.prepare_metadata(str(self.raw_root))
        super().__init__(
            root=str(metadata_path.parent),
            tables=["eda"],
            dataset_name=dataset_name or "wesad",
            config_path=config_path,
            cache_dir=cache_dir,
        )

    def prepare_metadata(self, root: str) -> Path:
        """Write one metadata row per participant recording in ``WESAD.zip``.

        Args:
            root: Directory containing the archive.

        Returns:
            Path to the metadata CSV.

        Raises:
            FileNotFoundError: The archive does not exist.
            zipfile.BadZipFile: The archive is not a valid ZIP file.
            ValueError: No supported recordings exist or a participant repeats.

        Examples:
            >>> dataset.prepare_metadata("data/wesad")  # doctest: +SKIP
        """
        root_path = Path(root).expanduser().resolve()
        archive_path = (root_path / "WESAD.zip").resolve()
        records = {}
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                match = _RECORDING.fullmatch(member.filename)
                if member.is_dir() or match is None:
                    continue
                subject = match.group("subject")
                if subject in records:
                    raise ValueError(f"Duplicate WESAD recording for {subject}.")
                records[subject] = {
                    "subject_id": subject,
                    "record_id": subject,
                    "archive_path": str(archive_path),
                    "member_path": member.filename,
                }
        if not records:
            raise ValueError(
                "No WESAD/S<number>/S<number>.pkl recordings found in WESAD.zip."
            )
        subjects = sorted(records, key=lambda subject: int(subject[1:]))
        metadata = pd.DataFrame([records[subject] for subject in subjects])
        return write_metadata_csv(metadata, root_path, "wesad", _CSV_NAME)
