from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reusable training experiments for the breast cancer analysis project.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an experiment JSON config. Defaults to configs/experiment.default.json.",
    )
    parser.add_argument("--raw-data-dir", default=None, help="Optional raw data directory. Only used to resolve default project paths.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--train-split", default=None, help="Override train split CSV path.")
    parser.add_argument("--test-split", default=None, help="Override test split CSV path.")
    parser.add_argument("--history-dir", default=None, help="Override training history output directory.")
    parser.add_argument("--predictions-dir", default=None, help="Override prediction output directory.")
    parser.add_argument("--folds", type=int, default=None, help="Use 0 for fixed train/test evaluation.")
    parser.add_argument("--validation-size", type=float, default=None, help="Validation fraction drawn from the train split.")
    parser.add_argument("--random-state", type=int, default=None, help="Seed used for split reproducibility and model initialization.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to run.")
    parser.add_argument("--preprocessing", nargs="*", default=None, help="Optional subset of preprocessing ids to run.")
    parser.add_argument(
        "--combined-preprocessing",
        dest="include_combinations",
        action="store_true",
        default=None,
        help="Explicitly enable pairwise preprocessing combinations.",
    )
    parser.add_argument(
        "--no-combined-preprocessing",
        dest="include_combinations",
        action="store_false",
        help="Disable pairwise preprocessing combinations.",
    )
    parser.add_argument(
        "--run-skip",
        dest="run_skip",
        action="store_true",
        default=None,
        help="Skip runs when history and prediction artifacts already exist.",
    )
    parser.add_argument(
        "--no-run-skip",
        dest="run_skip",
        action="store_false",
        help="Always rerun experiments even when artifacts already exist.",
    )
    return parser


def build_training_config_from_args(args: argparse.Namespace):
    from pipeline.config import load_experiment_config

    experiment_config = load_experiment_config(args.config)
    project_paths = experiment_config.resolve_project_paths(
        raw_data_dir=args.raw_data_dir,
        artifacts_dir=args.artifacts_dir,
    ).ensure_artifact_dirs()
    return experiment_config.build_training_config(
        project_paths,
        train_split=args.train_split,
        test_split=args.test_split,
        history_dir=args.history_dir,
        predictions_dir=args.predictions_dir,
        folds=args.folds,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss=args.loss,
        random_state=args.random_state,
        model_names=args.models,
        preprocessing_ids=args.preprocessing,
        include_combinations=args.include_combinations,
        run_skip=args.run_skip,
    )


def main() -> int:
    from pipeline.train.runner import run_training_pipeline

    args = build_parser().parse_args()
    config = build_training_config_from_args(args)
    summaries = run_training_pipeline(config)
    print(f"Completed {len(summaries)} experiment runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
