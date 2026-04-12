from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reusable training experiments for the breast cancer analysis project.")
    parser.add_argument("--raw-data-dir", default=None, help="Optional raw data directory. Only used to resolve default project paths.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--train-split", default=None, help="Override train split CSV path.")
    parser.add_argument("--test-split", default=None, help="Override test split CSV path.")
    parser.add_argument("--history-dir", default=None, help="Override training history output directory.")
    parser.add_argument("--predictions-dir", default=None, help="Override prediction output directory.")
    parser.add_argument("--folds", type=int, default=4, help="Use 0 for fixed train/test evaluation.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names to run.")
    parser.add_argument("--preprocessing", nargs="*", default=None, help="Optional subset of preprocessing ids to run.")
    parser.add_argument("--no-combined-preprocessing", action="store_true", help="Disable pairwise preprocessing combinations.")
    parser.add_argument("--no-run-skip", action="store_true", help="Always rerun experiments even when artifacts already exist.")
    return parser


def main() -> int:
    from pipeline.train.runner import TrainingConfig, run_training_pipeline
    from pipeline.utils.paths import build_project_paths

    args = build_parser().parse_args()
    paths = build_project_paths(args.raw_data_dir, args.artifacts_dir).ensure_artifact_dirs()
    config = TrainingConfig(
        train_split_path=paths.train_split_path if args.train_split is None else paths.project_root / args.train_split,
        test_split_path=paths.test_split_path if args.test_split is None else paths.project_root / args.test_split,
        history_dir=paths.history_dir if args.history_dir is None else paths.project_root / args.history_dir,
        predictions_dir=paths.predictions_dir if args.predictions_dir is None else paths.project_root / args.predictions_dir,
        folds=args.folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        run_skip=not args.no_run_skip,
        model_names=args.models,
        preprocessing_ids=args.preprocessing,
        include_combinations=not args.no_combined_preprocessing,
    )
    summaries = run_training_pipeline(config)
    print(f"Completed {len(summaries)} experiment runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
