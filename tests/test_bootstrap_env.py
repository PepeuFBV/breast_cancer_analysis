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
    assert plan.require_gpu_check is True
    assert plan.gpu_driver_visible is True


def test_resolve_bootstrap_plan_falls_back_to_cpu_when_driver_missing() -> None:
    plan = bootstrap_env.resolve_bootstrap_plan(
        "auto",
        platform_name="linux",
        driver_visible=False,
    )

    assert plan.tensorflow_requirement == "tensorflow"
    assert plan.require_gpu_check is False


def test_resolve_bootstrap_plan_requires_linux_gpu_driver() -> None:
    with pytest.raises(bootstrap_env.BootstrapError, match="No NVIDIA GPU driver"):
        bootstrap_env.resolve_bootstrap_plan(
            "required",
            platform_name="linux",
            driver_visible=False,
        )


def test_resolve_bootstrap_plan_rejects_required_gpu_on_non_linux() -> None:
    with pytest.raises(
        bootstrap_env.BootstrapError,
        match="only supported on Linux/WSL2",
    ):
        bootstrap_env.resolve_bootstrap_plan(
            "required",
            platform_name="darwin",
            driver_visible=True,
        )
