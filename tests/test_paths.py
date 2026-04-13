from __future__ import annotations

import unittest
from pathlib import Path

from pipeline.utils.paths import PROJECT_ROOT, build_project_paths


class ProjectPathsTest(unittest.TestCase):
    def test_default_layout_points_to_artifacts_structure(self) -> None:
        paths = build_project_paths()

        self.assertEqual(paths.project_root, PROJECT_ROOT)
        self.assertEqual(
            paths.raw_data_dir, PROJECT_ROOT / "data" / "INbreast Release 1.0"
        )
        self.assertEqual(
            paths.train_split_path,
            PROJECT_ROOT / "artifacts" / "processed" / "splits" / "train_split.csv",
        )
        self.assertEqual(
            paths.final_report_path,
            PROJECT_ROOT / "artifacts" / "reports" / "final_comprehensive_results.csv",
        )

    def test_custom_roots_are_respected(self) -> None:
        paths = build_project_paths("/tmp/raw-data", "/tmp/custom-artifacts")

        self.assertEqual(paths.raw_data_dir, Path("/tmp/raw-data"))
        self.assertEqual(paths.artifacts_dir, Path("/tmp/custom-artifacts"))
        self.assertEqual(paths.history_dir, Path("/tmp/custom-artifacts/runs/history"))


if __name__ == "__main__":
    unittest.main()
