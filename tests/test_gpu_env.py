from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.utils.gpu_env import (
    _nvidia_lib_dirs,
    _prepend_env_paths,
    _venv_root,
    ensure_tensorflow_wsl_gpu_env,
)


class GpuEnvHelperTest(unittest.TestCase):
    def test_venv_root_prefers_virtual_env_environment_variable(self) -> None:
        with patch.dict("os.environ", {"VIRTUAL_ENV": "/tmp/custom-venv"}, clear=False):
            self.assertEqual(_venv_root(), Path("/tmp/custom-venv"))

    def test_venv_root_uses_python_launcher_path_without_resolving_symlink(
        self,
    ) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("sys.executable", "/tmp/project/.venv/bin/python"):
                self.assertEqual(_venv_root(), Path("/tmp/project/.venv"))

    def test_nvidia_lib_dirs_discovers_all_cuda_library_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            site_packages = Path(tmp_dir) / "site-packages"
            (site_packages / "nvidia" / "cublas" / "lib").mkdir(parents=True)
            (site_packages / "nvidia" / "cudnn" / "lib").mkdir(parents=True)
            (site_packages / "nvidia" / "cusolver" / "lib64").mkdir(parents=True)
            (site_packages / "nvidia" / "cuda_runtime").mkdir(parents=True)

            discovered = _nvidia_lib_dirs(site_packages)

            self.assertEqual(
                discovered,
                [
                    site_packages / "nvidia" / "cublas" / "lib",
                    site_packages / "nvidia" / "cudnn" / "lib",
                    site_packages / "nvidia" / "cusolver" / "lib64",
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

    def test_wsl_driver_dir_is_bootstrapped_even_without_pip_cuda_libs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            venv_root = Path(tmp_dir) / ".venv"
            site_packages = venv_root / "lib" / "python3.12" / "site-packages"
            site_packages.mkdir(parents=True)

            with patch("pipeline.utils.gpu_env._is_wsl_linux", return_value=True):
                with patch(
                    "pipeline.utils.gpu_env._venv_root",
                    return_value=venv_root,
                ):
                    with patch(
                        "pipeline.utils.gpu_env._python_site_packages_dir",
                        return_value=site_packages,
                    ):
                        with patch(
                            "pipeline.utils.gpu_env._nvidia_lib_dirs",
                            return_value=[],
                        ):
                            with patch(
                                "pipeline.utils.gpu_env.Path.exists",
                                return_value=True,
                            ):
                                with patch(
                                    "pipeline.utils.gpu_env.os.execvpe"
                                ) as execvpe:
                                    with patch.dict("os.environ", {}, clear=True):
                                        with patch("sys.argv", ["check_gpu.py"]):
                                            ensure_tensorflow_wsl_gpu_env()

            execvpe.assert_called_once()
            _, _, environment = execvpe.call_args[0]
            self.assertIn("/usr/lib/wsl/lib", environment["PATH"])
            self.assertIn("/usr/lib/wsl/lib", environment["LD_LIBRARY_PATH"])
            self.assertEqual(
                environment["BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"],
                "1",
            )


if __name__ == "__main__":
    unittest.main()
