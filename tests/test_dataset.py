from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.data.dataset import (
    annotate_split_groups,
    balance_df_trim_above,
    load_metadata,
    normalize_file_number,
    split_dataset,
)


class DatasetHelpersTest(unittest.TestCase):
    def test_normalize_file_number_handles_numeric_formats(self) -> None:
        self.assertEqual(normalize_file_number("0007"), "7")
        self.assertEqual(normalize_file_number("20586908.0"), "20586908")
        self.assertEqual(normalize_file_number(12), "12")

    def test_load_metadata_filters_invalid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "INbreast.csv"
            pd.DataFrame(
                {
                    "File Name": ["0100", "101", "102.0"],
                    "Bi-Rads": ["1", "7", "4A"],
                }
            ).to_csv(csv_path, sep=";", index=False)

            metadata = load_metadata(csv_path)

            self.assertEqual(metadata["File Name"].tolist(), ["100", "102"])
            self.assertEqual(metadata["Bi-Rads"].tolist(), ["1", "4a"])

    def test_annotate_split_groups_prefers_patient_id(self) -> None:
        metadata = pd.DataFrame(
            {
                "File Name": ["100", "101"],
                "Bi-Rads": ["1", "2"],
                "Patient ID": ["P-01", "P-02"],
                "Laterality": ["L", "R"],
            }
        )

        annotated, strategy, group_columns = annotate_split_groups(metadata)

        self.assertEqual(strategy, "patient")
        self.assertEqual(group_columns, ("Patient ID",))
        self.assertEqual(annotated["split_group_id"].tolist(), ["P-01", "P-02"])

    def test_balance_and_split_preserve_group_boundaries(self) -> None:
        dataframe = pd.DataFrame(
            {
                "image_path": [f"img_{index}.png" for index in range(8)],
                "label": ["1", "1", "1", "1", "2", "2", "2", "2"],
                "split_group_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            }
        )

        balanced = balance_df_trim_above(dataframe, samples_per_class=4, random_state=42)
        split_result = split_dataset(balanced, test_size=0.5, random_state=42)
        train_groups = set(split_result.train_df["split_group_id"])
        test_groups = set(split_result.test_df["split_group_id"])

        self.assertEqual(len(balanced), 8)
        self.assertEqual(balanced["label"].value_counts().to_dict(), {"1": 4, "2": 4})
        self.assertEqual(sorted(split_result.train_df["label"].unique().tolist()), ["1", "2"])
        self.assertEqual(sorted(split_result.test_df["label"].unique().tolist()), ["1", "2"])
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertEqual(split_result.strategy, "stratified_group")


if __name__ == "__main__":
    unittest.main()
