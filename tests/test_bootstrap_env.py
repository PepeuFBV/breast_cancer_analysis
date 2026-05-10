from __future__ import annotations

import pytest

from scripts import bootstrap_env

pytestmark = pytest.mark.unit


def test_render_requirements_swaps_tensorflow_for_and_cuda() -> None:
    rendered = bootstrap_env.render_requirements(
        "numpy\ntensorflow\nkeras\n",
        "tensorflow[and-cuda]",
    )

    assert rendered.splitlines() == ["numpy", "tensorflow[and-cuda]", "keras"]


def test_render_requirements_adds_tensorflow_when_missing() -> None:
    rendered = bootstrap_env.render_requirements("numpy\nkeras\n", "tensorflow")

    assert rendered.splitlines() == ["numpy", "keras", "tensorflow"]


def test_resolve_bootstrap_plan_prefers_gpu_runtime_when_driver_visible() -> None:
    plan = bootstrap_env.resolve_bootstrap_plan(
        "auto",
        platform_name="linux",
        driver_visible=True,
    )

    assert plan.tensorflow_requirement == "tensorflow[and-cuda]"
    assert plan.requirements_path.name == "requirements.txt"
    assert plan.require_gpu_check is True
    assert plan.gpu_driver_visible is True
    assert plan.gpu_check_script == "check_gpu.py"


def test_resolve_bootstrap_plan_falls_back_to_cpu_when_driver_missing() -> None:
    plan = bootstrap_env.resolve_bootstrap_plan(
        "auto",
        platform_name="linux",
        driver_visible=False,
    )

    assert plan.tensorflow_requirement == "tensorflow"
    assert plan.requirements_path.name == "requirements.txt"
    assert plan.require_gpu_check is False
    assert plan.gpu_check_script == "check_gpu.py"


def test_resolve_bootstrap_plan_requires_linux_gpu_driver() -> None:
    with pytest.raises(bootstrap_env.BootstrapError, match="No NVIDIA GPU driver"):
        bootstrap_env.resolve_bootstrap_plan(
            "required",
            platform_name="linux",
            driver_visible=False,
        )


def test_resolve_bootstrap_plan_rejects_required_gpu_on_non_supported_platform() -> None:
    with pytest.raises(
        bootstrap_env.BootstrapError,
        match="only supported on Linux/WSL2 and native Windows",
    ):
        bootstrap_env.resolve_bootstrap_plan(
            "required",
            platform_name="darwin",
            driver_visible=True,
        )


def test_resolve_bootstrap_plan_windows_gpu_uses_single_venv_requirements() -> None:
    plan = bootstrap_env.resolve_bootstrap_plan(
        "auto",
        platform_name="win32",
        driver_visible=True,
    )

    assert plan.tensorflow_requirement is None
    assert plan.requirements_path.name == "requirements-windows-gpu.txt"
    assert plan.require_gpu_check is True
    assert plan.gpu_check_script == "check_windows_gpu.py"


def test_resolve_bootstrap_plan_windows_required_gpu_needs_driver() -> None:
    with pytest.raises(bootstrap_env.BootstrapError, match="No NVIDIA GPU driver is visible on native Windows"):
        bootstrap_env.resolve_bootstrap_plan(
            "required",
            platform_name="win32",
            driver_visible=False,
        )
