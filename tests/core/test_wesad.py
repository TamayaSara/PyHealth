"""Synthetic archive tests for the WESAD recording index."""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import yaml

from pyhealth.datasets.wesad import WESADDataset


class TestWESADDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cache = self.root / "processed"

    def make_archive(self, members):
        with (
            ZipFile(self.root / "WESAD.zip", "w") as archive,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", UserWarning)
            for member in members:
                archive.writestr(member, b"synthetic recording placeholder")

    def dataset(self, **kwargs):
        return WESADDataset(str(self.root), cache_dir=self.cache, **kwargs)

    def test_metadata_and_patient_events(self):
        self.make_archive(
            [
                "WESAD/S10/S10.pkl",
                "WESAD/S2/S2.pkl",
                "WESAD/readme.txt",
                "__MACOSX/WESAD/S3/S3.pkl",
                "WESAD/S4/S5.pkl",
                "elsewhere/S6/S6.pkl",
                "WESAD/S7/S7.pkl/",
            ]
        )
        with patch.object(ZipFile, "open", side_effect=AssertionError("read signal")):
            dataset = self.dataset()
        metadata = pd.read_csv(self.root / "wesad-pyhealth.csv")
        self.assertEqual(metadata.subject_id.tolist(), ["S2", "S10"])
        self.assertEqual(metadata.record_id.tolist(), ["S2", "S10"])
        self.assertEqual(set(dataset.unique_patient_ids), {"S2", "S10"})
        events = dataset.get_patient("S2").get_events("eda")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["member_path"], "WESAD/S2/S2.pkl")
        self.assertEqual(events[0]["record_id"], "S2")
        self.assertEqual(
            events[0]["archive_path"], str((self.root / "WESAD.zip").resolve())
        )

    def test_duplicate_recording(self):
        self.make_archive(["WESAD/S2/S2.pkl", "WESAD/S2/S2.pkl"])
        with self.assertRaisesRegex(ValueError, "Duplicate.*S2"):
            self.dataset()
        self.assertFalse((self.root / "wesad-pyhealth.csv").exists())

    def test_no_recordings(self):
        self.make_archive(["WESAD/readme.txt", "WESAD/S2/S3.pkl"])
        with self.assertRaisesRegex(ValueError, "No WESAD"):
            self.dataset()

    def test_replaced_archive_refreshes_metadata(self):
        self.make_archive(["WESAD/S2/S2.pkl"])
        first = self.dataset()
        self.assertEqual(first.unique_patient_ids, ["S2"])
        self.make_archive(["WESAD/S3/S3.pkl", "WESAD/S4/S4.pkl"])
        second = self.dataset()
        self.assertEqual(set(second.unique_patient_ids), {"S3", "S4"})
        self.assertNotEqual(first.cache_dir, second.cache_dir)

    def test_unwritable_root_uses_fresh_fallback(self):
        self.make_archive(["WESAD/S2/S2.pkl"])
        self.dataset()
        self.make_archive(["WESAD/S3/S3.pkl"])
        fallback = self.root / "metadata-cache"
        original = pd.DataFrame.to_csv

        def write(frame, path, *args, **kwargs):
            if Path(path).parent == self.root:
                raise PermissionError("synthetic read-only root")
            return original(frame, path, *args, **kwargs)

        with (
            patch(
                "pyhealth.datasets.metadata_csv.metadata_cache_dir",
                return_value=fallback,
            ),
            patch.object(pd.DataFrame, "to_csv", write),
        ):
            dataset = self.dataset()
        self.assertEqual(Path(dataset.root), fallback)
        self.assertEqual(dataset.unique_patient_ids, ["S3"])
        self.assertEqual(
            pd.read_csv(self.root / "wesad-pyhealth.csv").subject_id.tolist(), ["S2"]
        )

    def test_custom_config_and_name(self):
        self.make_archive(["WESAD/S2/S2.pkl"])
        config_path = Path(__file__).parents[2] / "pyhealth/datasets/configs/wesad.yaml"
        custom = self.root / "config.yaml"
        config = yaml.safe_load(config_path.read_text())
        custom.write_text(yaml.safe_dump(config))
        dataset = self.dataset(config_path=str(custom), dataset_name="custom-wesad")
        self.assertEqual(dataset.dataset_name, "custom-wesad")
        config["tables"]["eda"]["file_path"] = "wrong.csv"
        custom.write_text(yaml.safe_dump(config))
        with self.assertRaisesRegex(ValueError, "metadata mapping"):
            self.dataset(config_path=str(custom))


if __name__ == "__main__":
    unittest.main()
