"""Synthetic archive tests for WESAD wrist EDA stress classification."""

import gc
import pickle
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from litdata.streaming.shuffle import NoShuffle, Shuffle

from pyhealth.datasets import WESADDataset
from pyhealth.tasks import WESADStressClassification
from pyhealth.tasks.wesad import EDA_SAMPLING_RATE, LABEL_SAMPLING_RATE


class TestWESADStressClassification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "processed"

    def dataset(self, eda, labels, *, legacy_pickle=False):
        member_path = "WESAD/S2/S2.pkl"
        recording = {
            "signal": {"wrist": {"EDA": np.asarray(eda, dtype=np.float32)}},
            "label": np.asarray(labels, dtype=np.int64),
        }
        payload = pickle.dumps(recording, protocol=2 if legacy_pickle else None)
        if legacy_pickle:
            # Insert a Python 2 short string with a non-ASCII byte before STOP.
            payload = payload[:-1] + b"U\x04noteU\x01\xa3s."
        with ZipFile(self.root / "WESAD.zip", "w") as archive:
            archive.writestr(member_path, payload)
        return WESADDataset(str(self.root), cache_dir=self.cache)

    def test_sampling_rates_label_codes_and_archive_alignment(self):
        self.assertEqual(EDA_SAMPLING_RATE, 4)
        self.assertEqual(LABEL_SAMPLING_RATE, 700)
        self.assertEqual(LABEL_SAMPLING_RATE // EDA_SAMPLING_RATE, 175)

        eda = np.arange(24, dtype=np.float32).reshape(-1, 1)
        labels = np.concatenate(
            [
                np.full(1400, 1),
                np.full(700, 1),
                np.full(700, 2),
                np.full(1400, 2),
            ]
        )
        dataset = self.dataset(eda, labels, legacy_pickle=True)
        patient = dataset.get_patient("S2")
        samples = WESADStressClassification(window_seconds=2)(patient)

        self.assertEqual(
            [sample["label"] for sample in samples], ["baseline", "stress"]
        )
        self.assertEqual([sample["window_index"] for sample in samples], [0, 2])
        self.assertEqual([sample["start_seconds"] for sample in samples], [0, 4])
        self.assertEqual(tuple(samples[0]["signal"].shape), (1, 8))
        self.assertTrue(
            torch.equal(samples[0]["signal"], torch.arange(8).float()[None])
        )
        self.assertTrue(
            torch.equal(samples[1]["signal"], torch.arange(16, 24).float()[None])
        )
        self.assertEqual(samples[0]["record_id"], "S2")

    def test_set_task_processes_archive_backed_samples_at_two_window_lengths(self):
        eda = np.arange(48, dtype=np.float32)
        labels = np.concatenate(
            [
                np.full(4200, 1),
                np.full(4200, 2),
            ]
        )
        dataset = self.dataset(eda, labels)
        # LitData's shuffle caches retain readers from the intermediate task data.
        # Cleanups run in reverse order, before TemporaryDirectory.cleanup.
        self.addCleanup(gc.collect)
        self.addCleanup(Shuffle.get_len.cache_clear)
        self.addCleanup(NoShuffle.get_chunks_and_intervals_per_workers.cache_clear)
        for window_seconds, expected_count, expected_samples in (
            (2, 6, 8),
            (3, 4, 12),
        ):
            with self.subTest(window_seconds=window_seconds):
                with dataset.set_task(
                    WESADStressClassification(window_seconds=window_seconds),
                    num_workers=1,
                ) as samples:
                    try:
                        self.assertEqual(len(samples), expected_count)
                        self.assertEqual(
                            tuple(samples[0]["signal"].shape), (1, expected_samples)
                        )
                        label_processor = samples.output_processors["label"]
                        self.assertEqual(
                            set(label_processor.label_vocab), {"baseline", "stress"}
                        )
                    finally:
                        samples.reset()

    def test_discards_mixed_other_and_incomplete_windows(self):
        eda = np.arange(24, dtype=np.float32)
        labels = np.concatenate(
            [
                np.full(1400, 1),
                np.full(1400, 3),
                np.full(1399, 2),
            ]
        )
        samples = WESADStressClassification(window_seconds=2)(
            self.dataset(eda, labels).get_patient("S2")
        )

        self.assertEqual([sample["label"] for sample in samples], ["baseline"])

    def test_rejects_mismatched_recording_durations(self):
        dataset = self.dataset(np.arange(24), np.ones(3500, dtype=np.int64))

        with self.assertRaisesRegex(ValueError, "mismatched EDA and label durations"):
            WESADStressClassification(window_seconds=2)(
                dataset.get_patient("S2")
            )

    def test_rejects_unknown_label_codes(self):
        labels = np.ones(4200, dtype=np.int64)
        labels[-1] = 8
        dataset = self.dataset(np.arange(24), labels)

        with self.assertRaisesRegex(ValueError, "unsupported codes: \\[8\\]"):
            WESADStressClassification(window_seconds=2)(
                dataset.get_patient("S2")
            )

    def test_window_duration_must_align_to_eda_samples(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            WESADStressClassification(window_seconds=0.1)
        with self.assertRaisesRegex(ValueError, "whole number"):
            WESADStressClassification(window_seconds=1e-10)


if __name__ == "__main__":
    unittest.main()
