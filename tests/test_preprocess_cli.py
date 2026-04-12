from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from preprocess import validate_raw_dataset_layout


class PreprocessCliValidationTest(unittest.TestCase):
    def test_validate_raw_dataset_layout_reports_missing_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_dir = Path(tmp_dir) / "INbreast Release 1.0"
            dicom_dir = raw_dir / "AllDICOMs"
            dicom_dir.mkdir(parents=True)
            (dicom_dir / "sample.dcm").write_bytes(b"fake")

            with self.assertRaises(FileNotFoundError) as context:
                validate_raw_dataset_layout(raw_dir)

            self.assertIn("metadata CSV", str(context.exception))
            self.assertIn("INbreast.csv", str(context.exception))


if __name__ == "__main__":
    unittest.main()
