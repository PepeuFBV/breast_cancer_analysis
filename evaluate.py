from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation reports from reusable training artifacts.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to an experiment JSON config. Defaults to configs/experiment.default.json.",
    )
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--history-dir", default=None, help="Override history directory.")
    parser.add_argument("--predictions-dir", default=None, help="Override predictions directory.")
    parser.add_argument("--output-path", default=None, help="Override final report CSV output path.")
    parser.add_argument("--top-k", type=int, default=None)
    return parser


def build_evaluation_config_from_args(args: argparse.Namespace):
    from pipeline.config import load_experiment_config

    experiment_config = load_experiment_config(args.config)
    project_paths = experiment_config.resolve_project_paths(artifacts_dir=args.artifacts_dir).ensure_artifact_dirs()
    return experiment_config.build_evaluation_config(
        project_paths,
        history_dir=args.history_dir,
        predictions_dir=args.predictions_dir,
        output_path=args.output_path,
        top_k=args.top_k,
    )


def main() -> int:
    from pipeline.evaluate.reporting import generate_final_report

    args = build_parser().parse_args()
    config = build_evaluation_config_from_args(args)
    final_results, output_path = generate_final_report(config)
    print(f"Saved final report to: {output_path}")
    print(f"Records: {len(final_results)}")
    print(f"Preprocessing methods: {final_results['preproc_id'].nunique()}")
    print(f"Models: {final_results['model_name'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
