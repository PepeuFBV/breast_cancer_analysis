from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import load_experiment_config
from pipeline.data.validation import (
    format_dataset_layout_report,
    inspect_dataset_layout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the expected INbreast dataset directory layout.")
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment config path. Defaults to configs/experiment.default.json.",
    )
    parser.add_argument(
        "--raw-data-dir",
        default=None,
        help="Raw dataset directory. Overrides the config path.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Artifacts root used only to resolve project paths from the config.",
    )
    parser.add_argument(
        "--min-dicoms",
        type=int,
        default=1,
        help="Minimum number of .dcm files required under AllDICOMs/.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_config = load_experiment_config(args.config)
    project_paths = experiment_config.resolve_project_paths(
        raw_data_dir=args.raw_data_dir,
        artifacts_dir=args.artifacts_dir,
    )
    result = inspect_dataset_layout(
        project_paths.raw_data_dir,
        min_dicoms=args.min_dicoms,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "raw_data_dir": str(result.raw_data_dir),
                    "metadata_path": str(result.metadata_path),
                    "dicom_dir": str(result.dicom_dir),
                    "dicom_count": result.dicom_count,
                    "errors": list(result.errors),
                    "warnings": list(result.warnings),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_dataset_layout_report(result))

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
