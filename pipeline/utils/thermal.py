from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_WSL_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")


@dataclass(frozen=True)
class ThermalGpuSample:
    name: str
    temperature_celsius: float | None
    utilization_percent: float | None


@dataclass(frozen=True)
class ThermalSnapshot:
    cpu_temperature_celsius: float | None
    cpu_load_percent: float | None
    gpu_samples: tuple[ThermalGpuSample, ...]
    max_gpu_temperature_celsius: float | None
    max_gpu_utilization_percent: float | None
    warnings: tuple[str, ...]


def _nvidia_smi_command() -> list[str] | None:
    if _WSL_NVIDIA_SMI.exists():
        return [str(_WSL_NVIDIA_SMI)]
    executable = shutil.which("nvidia-smi")
    if executable:
        return [executable]
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw or raw in {"N/A", "[Not Supported]"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_cpu_temperature_celsius() -> float | None:
    try:
        import psutil  # type: ignore

        sensors = psutil.sensors_temperatures()
        for entries in sensors.values():
            for entry in entries:
                current = getattr(entry, "current", None)
                if isinstance(current, (int, float)):
                    return float(current)
    except Exception:
        pass

    for temp_path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = temp_path.read_text(encoding="utf-8").strip()
            value = float(raw)
        except Exception:
            continue
        if value > 1_000:
            value /= 1_000.0
        if value > 0:
            return value
    return None


def _read_cpu_load_percent() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass

    if not hasattr(os, "getloadavg"):
        return None
    try:
        load_one_minute = float(os.getloadavg()[0])
    except Exception:
        return None
    cpu_count = os.cpu_count() or 1
    return max(0.0, load_one_minute / float(cpu_count) * 100.0)


def collect_thermal_snapshot() -> ThermalSnapshot:
    warnings: list[str] = []
    cpu_temperature_celsius = _read_cpu_temperature_celsius()
    cpu_load_percent = _read_cpu_load_percent()
    gpu_samples: list[ThermalGpuSample] = []

    command = _nvidia_smi_command()
    if command:
        query = [
            *command,
            "--query-gpu=name,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                query,
                check=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                parts = [item.strip() for item in line.split(",")]
                if len(parts) != 3:
                    continue
                gpu_samples.append(
                    ThermalGpuSample(
                        name=parts[0],
                        temperature_celsius=_parse_float(parts[1]),
                        utilization_percent=_parse_float(parts[2]),
                    )
                )
        except Exception as error:
            warnings.append(f"Thermal GPU probe failed: {error}")

    max_gpu_temperature_celsius = max(
        (sample.temperature_celsius for sample in gpu_samples if sample.temperature_celsius is not None),
        default=None,
    )
    max_gpu_utilization_percent = max(
        (sample.utilization_percent for sample in gpu_samples if sample.utilization_percent is not None),
        default=None,
    )

    return ThermalSnapshot(
        cpu_temperature_celsius=cpu_temperature_celsius,
        cpu_load_percent=cpu_load_percent,
        gpu_samples=tuple(gpu_samples),
        max_gpu_temperature_celsius=max_gpu_temperature_celsius,
        max_gpu_utilization_percent=max_gpu_utilization_percent,
        warnings=tuple(warnings),
    )
