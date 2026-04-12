from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.data.dataset import balance_df_trim_above, load_metadata, normalize_file_number, split_dataset


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

    def test_balance_and_split_preserve_class_distribution_shape(self) -> None:
        dataframe = pd.DataFrame(
            {
                "image_path": [f"img_{index}.png" for index in range(14)],
                "label": ["1"] * 8 + ["2"] * 6,
            }
        )

        balanced = balance_df_trim_above(dataframe, samples_per_class=4, random_state=42)
        train_df, test_df = split_dataset(balanced, test_size=0.25, random_state=42)

        self.assertEqual(len(balanced), 8)
        self.assertEqual(balanced["label"].value_counts().to_dict(), {"1": 4, "2": 4})
        self.assertEqual(sorted(train_df["label"].unique().tolist()), ["1", "2"])
        self.assertEqual(sorted(test_df["label"].unique().tolist()), ["1", "2"])


if __name__ == "__main__":
    unittest.main()
