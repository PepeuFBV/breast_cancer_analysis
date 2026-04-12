from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evaluation reports from reusable training artifacts.")
    parser.add_argument("--artifacts-dir", default=None, help="Artifacts root directory. Defaults to artifacts/.")
    parser.add_argument("--history-dir", default=None, help="Override history directory.")
    parser.add_argument("--predictions-dir", default=None, help="Override predictions directory.")
    parser.add_argument("--output-path", default=None, help="Override final report CSV output path.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> int:
    from pipeline.evaluate.reporting import EvaluationConfig, generate_final_report
    from pipeline.utils.paths import build_project_paths

    args = build_parser().parse_args()
    paths = build_project_paths(artifacts_dir=args.artifacts_dir).ensure_artifact_dirs()
    config = EvaluationConfig(
        history_dir=paths.history_dir if args.history_dir is None else paths.project_root / args.history_dir,
        predictions_dir=paths.predictions_dir if args.predictions_dir is None else paths.project_root / args.predictions_dir,
        output_path=paths.final_report_path if args.output_path is None else paths.project_root / args.output_path,
        top_k=args.top_k,
    )
    final_results, output_path = generate_final_report(config)
    print(f"Saved final report to: {output_path}")
    print(f"Records: {len(final_results)}")
    print(f"Preprocessing methods: {final_results['preproc_id'].nunique()}")
    print(f"Models: {final_results['model_name'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
