from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline.evaluate.reporting import EvaluationConfig, generate_final_report
from pipeline.utils.naming import param_dict_to_display, param_dict_to_file_id, param_dict_to_json


class ReportingTest(unittest.TestCase):
    def test_param_serialization_helpers(self) -> None:
        params = {"binarize_params": {"method": "fixed", "threshold": 70}, "denoise_params": {"kernel_size": (3, 3), "sigma": 0}}

        self.assertIn("binarize_params", param_dict_to_display(params))
        self.assertEqual(json.loads(param_dict_to_json(params))["denoise_params"]["kernel_size"], [3, 3])
        self.assertTrue(param_dict_to_file_id(params).startswith("params-"))

    def test_generate_final_report_merges_history_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            history_dir = root / "history" / "none" / "custom cnn"
            predictions_dir = root / "predictions" / "none" / "custom cnn"
            history_dir.mkdir(parents=True)
            predictions_dir.mkdir(parents=True)

            params = {"threshold": 70}
            param_id = param_dict_to_file_id(params)
            history_df = pd.DataFrame(
                [
                    {
                        "fold": 1,
                        "best_val_acc": 0.9,
                        "best_epoch": 3,
                        "preproc_id": "none",
                        "model_name": "custom cnn",
                        "param_id": param_id,
                        "param_combo": param_dict_to_display(params),
                        "param_json": param_dict_to_json(params),
                        "accuracy": 0.8,
                        "val_accuracy": 0.9,
                    }
                ]
            )
            history_df.to_csv(history_dir / f"history_{param_id}.csv", index=False)

            predictions_df = pd.DataFrame(
                [
                    {"y_true": 0, "y_pred": 0, "prob_class_0": 0.9, "prob_class_1": 0.1},
                    {"y_true": 1, "y_pred": 1, "prob_class_0": 0.2, "prob_class_1": 0.8},
                ]
            )
            predictions_df.to_csv(predictions_dir / f"{param_id}.csv", index=False)

            config = EvaluationConfig(
                history_dir=root / "history",
                predictions_dir=root / "predictions",
                output_path=root / "reports" / "final.csv",
            )
            final_results, output_path = generate_final_report(config)

            self.assertTrue(output_path.exists())
            self.assertEqual(len(final_results), 1)
            self.assertIn("top_k_accuracy", final_results.columns)
            self.assertIn("f1_macro", final_results.columns)
            self.assertIn("roc_auc_ovr_macro", final_results.columns)
            self.assertEqual(final_results.iloc[0]["threshold"], 70)
            self.assertTrue(Path(final_results.iloc[0]["classification_report_path"]).exists())
            self.assertTrue(Path(final_results.iloc[0]["confusion_matrix_path"]).exists())
            self.assertTrue(Path(final_results.iloc[0]["metrics_summary_path"]).exists())


if __name__ == "__main__":
    unittest.main()
