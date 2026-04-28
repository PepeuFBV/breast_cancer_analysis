from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate import build_evaluation_config_from_args
from evaluate import build_parser as build_evaluate_parser
from pipeline.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from preprocess import build_dataset_config_from_args
from preprocess import build_parser as build_preprocess_parser
from train import build_parser as build_train_parser
from train import build_training_config_from_args


class ExperimentConfigTest(unittest.TestCase):
    def test_load_default_config(self) -> None:
        config = load_experiment_config()

        self.assertEqual(config.source_path, DEFAULT_EXPERIMENT_CONFIG_PATH)
        self.assertEqual(config.preprocess.image_size, (224, 224))
        self.assertEqual(config.preprocess.random_state, 42)
        self.assertEqual(config.train.learning_rate, 1e-4)
        self.assertEqual(config.train.validation_size, 0.2)
        self.assertEqual(config.train.random_state, 42)
        self.assertFalse(config.train.include_combinations)
        self.assertIn("custom cnn", config.models)
        self.assertEqual(config.models["custom cnn"].input_channels, 1)

    def test_relative_paths_resolve_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "custom.json"
            payload = json.loads(
                DEFAULT_EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8")
            )
            payload["paths"]["artifacts_dir"] = "tmp/artifacts"
            payload["paths"]["history_dir"] = "tmp/history"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = load_experiment_config(config_path)

            self.assertTrue(str(config.paths.artifacts_dir).endswith("tmp/artifacts"))
            self.assertTrue(str(config.paths.history_dir).endswith("tmp/history"))

    def test_invalid_preprocessing_name_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid.json"
            payload = json.loads(
                DEFAULT_EXPERIMENT_CONFIG_PATH.read_text(encoding="utf-8")
            )
            payload["preprocess"]["preprocessing_grid"]["unknown-filter"] = {
                "alpha": [1]
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_experiment_config(config_path)

    def test_train_cli_uses_config_and_explicit_overrides(self) -> None:
        parser = build_train_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/experiment.default.json",
                "--epochs",
                "3",
                "--batch-size",
                "2",
                "--validation-size",
                "0.3",
                "--random-state",
                "99",
                "--no-combined-preprocessing",
            ]
        )

        config = build_training_config_from_args(args)

        self.assertEqual(config.epochs, 3)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.validation_size, 0.3)
        self.assertEqual(config.random_state, 99)
        self.assertFalse(config.include_combinations)
        self.assertEqual(config.learning_rate, 1e-4)

    def test_preprocess_cli_uses_config_defaults(self) -> None:
        parser = build_preprocess_parser()
        args = parser.parse_args(
            ["--config", "configs/experiment.default.json", "--random-state", "123"]
        )

        config = build_dataset_config_from_args(args)

        self.assertEqual(config.resize_dim, (224, 224))
        self.assertEqual(config.augmentations_per_image, 3)
        self.assertEqual(config.random_state, 123)

    def test_evaluate_cli_overrides_top_k(self) -> None:
        parser = build_evaluate_parser()
        args = parser.parse_args(
            [
                "--config",
                "configs/experiment.default.json",
                "--top-k",
                "5",
                "--details-dir",
                "artifacts/reports/custom-details",
            ]
        )

        config = build_evaluation_config_from_args(args)

        self.assertEqual(config.top_k, 5)
        self.assertTrue(
            str(config.details_dir).endswith("artifacts/reports/custom-details")
        )


if __name__ == "__main__":
    unittest.main()
