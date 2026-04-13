from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.utils.gpu_env import _nvidia_lib_dirs, _prepend_env_paths, _venv_root


class GpuEnvHelperTest(unittest.TestCase):
    def test_venv_root_prefers_virtual_env_environment_variable(self) -> None:
        with patch.dict("os.environ", {"VIRTUAL_ENV": "/tmp/custom-venv"}, clear=False):
            self.assertEqual(_venv_root(), Path("/tmp/custom-venv"))

    def test_venv_root_uses_python_launcher_path_without_resolving_symlink(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.executable", "/tmp/project/.venv/bin/python"):
                self.assertEqual(_venv_root(), Path("/tmp/project/.venv"))

    def test_nvidia_lib_dirs_discovers_all_cuda_library_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_packages = Path(tmp_dir) / "site-packages"
            (site_packages / "nvidia" / "cublas" / "lib").mkdir(parents=True)
            (site_packages / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
            (site_packages / "nvidia" / "cuda_runtime").mkdir(parents=True)

            discovered = _nvidia_lib_dirs(site_packages)

            self.assertEqual(
                discovered,
                [
                    site_packages / "nvidia" / "cublas" / "lib",
                    site_packages / "nvidia" / "cudnn" / "lib",
                ],
            )

    def test_prepend_env_paths_adds_unique_entries_once(self) -> None:
        environment = {"LD_LIBRARY_PATH": "/existing"}

        changed = _prepend_env_paths(
            environment,
            "LD_LIBRARY_PATH",
            [Path("/new-a"), Path("/existing"), Path("/new-b")],
        )

        self.assertTrue(changed)
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/new-a:/new-b:/existing")


if __name__ == "__main__":
    unittest.main()
