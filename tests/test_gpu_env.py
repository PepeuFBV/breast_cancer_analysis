from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.utils.gpu_env import (
    _nvidia_lib_dirs,
    _prepend_env_paths,
    _venv_root,
    bootstrap_tensorflow_runtime_env,
    ensure_native_windows_tensorflow_env,
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
        existing = str(Path("/existing"))
        new_a = str(Path("/new-a"))
        new_b = str(Path("/new-b"))
        environment = {"LD_LIBRARY_PATH": existing}

        changed = _prepend_env_paths(
            environment,
            "LD_LIBRARY_PATH",
            [Path("/new-a"), Path("/existing"), Path("/new-b")],
        )

        self.assertTrue(changed)
        self.assertEqual(
            environment["LD_LIBRARY_PATH"],
            os.pathsep.join([new_a, new_b, existing]),
        )

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
                                with patch("pipeline.utils.gpu_env.os.execvpe") as execvpe:
                                    with patch.dict("os.environ", {}, clear=True):
                                        with patch("sys.argv", ["check_gpu.py"]):
                                            ensure_tensorflow_wsl_gpu_env()

            execvpe.assert_called_once()
            _, _, environment = execvpe.call_args[0]
            wsl_driver_dir = str(Path("/usr/lib/wsl/lib"))
            self.assertIn(wsl_driver_dir, environment["PATH"])
            self.assertIn(wsl_driver_dir, environment["LD_LIBRARY_PATH"])
            self.assertEqual(
                environment["BREAST_CANCER_ANALYSIS_GPU_ENV_BOOTSTRAPPED"],
                "1",
            )

    def test_bootstrap_tensorflow_runtime_env_skips_non_wsl(self) -> None:
        with patch("pipeline.utils.gpu_env.should_bootstrap_wsl_tensorflow_env", return_value=False):
            with patch("pipeline.utils.gpu_env.ensure_tensorflow_wsl_gpu_env") as ensure_wsl:
                bootstrap_tensorflow_runtime_env()
        ensure_wsl.assert_not_called()

    def test_native_windows_bootstrap_prepends_cuda_112_dirs_for_tf210(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cuda_root = Path(tmp_dir) / "CUDA" / "v11.2"
            bin_dir = cuda_root / "bin"
            libnvvp_dir = cuda_root / "libnvvp"
            bin_dir.mkdir(parents=True)
            libnvvp_dir.mkdir(parents=True)
            (bin_dir / "cudart64_110.dll").write_text("", encoding="utf-8")

            with patch("pipeline.utils.gpu_env.is_native_windows", return_value=True):
                with patch(
                    "pipeline.utils.gpu_env._tensorflow_distribution_version",
                    return_value="2.10.1",
                ):
                    with patch(
                        "pipeline.utils.gpu_env._native_windows_cuda_root_candidates",
                        return_value=[cuda_root],
                    ):
                        with patch("pipeline.utils.gpu_env._WINDOWS_DLL_DIR_HANDLES", []):
                            with patch(
                                "pipeline.utils.gpu_env.os.add_dll_directory",
                                side_effect=lambda path: path,
                                create=True,
                            ) as add_dll_directory:
                                with patch.dict("os.environ", {"PATH": r"C:\existing"}, clear=True):
                                    ensure_native_windows_tensorflow_env()
                                    self.assertEqual(
                                        os.environ["BREAST_CANCER_ANALYSIS_WINDOWS_GPU_ENV_BOOTSTRAPPED"],
                                        "1",
                                    )
                                    self.assertTrue(
                                        os.environ["PATH"].startswith(
                                            str(bin_dir) + os.pathsep + str(libnvvp_dir)
                                        )
                                    )
                                    self.assertEqual(os.environ["CUDA_PATH"], str(cuda_root))
                                    self.assertEqual(os.environ["CUDA_PATH_V11_2"], str(cuda_root))

            self.assertEqual(add_dll_directory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
