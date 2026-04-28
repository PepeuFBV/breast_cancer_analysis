#!/usr/bin/env python3
"""Diagnose and fix GPU memory issues for the breast cancer analysis pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def check_gpu_memory() -> dict[str, any]:
    """Check current GPU memory usage."""
    try:
        result = subprocess.run(
            ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(","))
            return {"used_mb": used, "total_mb": total, "free_mb": total - used, "usage_percent": (used / total) * 100}
    except Exception:
        pass
    
    return {"error": "Could not query GPU memory"}


def count_failed_experiments() -> int:
    """Count failed experiments in the runner state."""
    state_file = PROJECT_ROOT / "artifacts" / "experiments" / "state" / "runner_state.json"
    if not state_file.exists():
        return 0
    
    try:
        with open(state_file) as f:
            state = json.load(f)
        return sum(1 for task in state.get("tasks", []) if task.get("status") == "failed")
    except Exception:
        return 0


def analyze_log_for_oom() -> dict[str, any]:
    """Analyze the runner log for OOM errors."""
    log_file = PROJECT_ROOT / "artifacts" / "experiments" / "logs" / "iterative-runner.log"
    if not log_file.exists():
        return {"oom_count": 0, "models_with_oom": []}
    
    oom_count = 0
    models_with_oom = set()
    
    try:
        with open(log_file) as f:
            for line in f:
                if "ResourceExhaustedError" in line or "OOM" in line or "failed to allocate memory" in line:
                    oom_count += 1
                    # Try to extract model name
                    if "[" in line and "]" in line:
                        parts = line.split("[")
                        if len(parts) > 1:
                            model_part = parts[1].split("]")[0]
                            if " - " in model_part:
                                model_name = model_part.split(" - ")[0].strip()
                                models_with_oom.add(model_name)
    except Exception:
        pass
    
    return {"oom_count": oom_count, "models_with_oom": sorted(models_with_oom)}


def suggest_fixes(gpu_info: dict, oom_info: dict) -> list[str]:
    """Suggest fixes based on the diagnosis."""
    suggestions = []
    
    if "error" in gpu_info:
        suggestions.append("GPU not accessible. Run on CPU with: python3 scripts/bootstrap_env.py --gpu off")
        return suggestions
    
    if gpu_info["total_mb"] <= 4096:
        suggestions.append(f"GPU has limited memory ({gpu_info['total_mb']} MB). Consider:")
        suggestions.append("  1. Reduce batch size in configs/experiment.default.json (try batch_size: 4 or 2)")
        suggestions.append("  2. Limit models to smaller ones: --models 'custom cnn' bcnet mobilenetv3")
        suggestions.append("  3. Run on CPU: python3 scripts/bootstrap_env.py --gpu off")
    
    if oom_info["oom_count"] > 0:
        suggestions.append(f"Found {oom_info['oom_count']} OOM errors in logs")
        if oom_info["models_with_oom"]:
            suggestions.append(f"  Models with OOM: {', '.join(oom_info['models_with_oom'])}")
        suggestions.append("  The pipeline now includes automatic retry with memory cleanup")
        suggestions.append("  Rerun failed experiments: ./.venv/bin/python run_experiments.py run --rerun-failed")
    
    if gpu_info["usage_percent"] > 80:
        suggestions.append(f"GPU memory is {gpu_info['usage_percent']:.1f}% full")
        suggestions.append("  Stop the runner and restart to clear memory:")
        suggestions.append("    ./.venv/bin/python run_experiments.py stop")
        suggestions.append("    Wait for current experiment to finish, then:")
        suggestions.append("    ./.venv/bin/python run_experiments.py run")
    
    return suggestions


def print_diagnosis(gpu_info: dict, oom_info: dict, failed_count: int) -> None:
    """Print diagnosis report."""
    print("=" * 70)
    print("GPU MEMORY DIAGNOSIS")
    print("=" * 70)
    
    print("\nGPU Memory Status:")
    if "error" in gpu_info:
        print(f"  ✗ {gpu_info['error']}")
    else:
        print(f"  Total: {gpu_info['total_mb']} MB")
        print(f"  Used: {gpu_info['used_mb']} MB ({gpu_info['usage_percent']:.1f}%)")
        print(f"  Free: {gpu_info['free_mb']} MB")
    
    print(f"\nExperiment Status:")
    print(f"  Failed experiments: {failed_count}")
    print(f"  OOM errors in log: {oom_info['oom_count']}")
    if oom_info["models_with_oom"]:
        print(f"  Models with OOM: {', '.join(oom_info['models_with_oom'])}")
    
    suggestions = suggest_fixes(gpu_info, oom_info)
    if suggestions:
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        for suggestion in suggestions:
            print(suggestion)
    
    print("\n" + "=" * 70)


def apply_fix(fix_type: str) -> int:
    """Apply a specific fix."""
    if fix_type == "reduce-batch":
        config_file = PROJECT_ROOT / "configs" / "experiment.default.json"
        if not config_file.exists():
            print(f"Config file not found: {config_file}")
            return 1
        
        with open(config_file) as f:
            config = json.load(f)
        
        old_batch = config["train"]["batch_size"]
        new_batch = max(2, old_batch // 2)
        config["train"]["batch_size"] = new_batch
        
        # Also reduce model-specific batch sizes
        for model_config in config.get("models", {}).values():
            if "batch_size" in model_config:
                model_config["batch_size"] = max(2, model_config["batch_size"] // 2)
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Reduced batch size from {old_batch} to {new_batch}")
        print(f"  Updated: {config_file}")
        return 0
    
    elif fix_type == "rerun-failed":
        result = subprocess.run(
            [str(VENV_PYTHON), "run_experiments.py", "run", "--rerun-failed"],
            cwd=PROJECT_ROOT,
        )
        return result.returncode
    
    elif fix_type == "stop-and-restart":
        print("Stopping runner...")
        subprocess.run([str(VENV_PYTHON), "run_experiments.py", "stop"], cwd=PROJECT_ROOT)
        print("\nWait for the current experiment to finish, then run:")
        print("  ./.venv/bin/python run_experiments.py run")
        return 0
    
    else:
        print(f"Unknown fix type: {fix_type}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose and fix GPU memory issues."
    )
    parser.add_argument(
        "--fix",
        choices=["reduce-batch", "rerun-failed", "stop-and-restart"],
        help="Apply a specific fix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    
    if args.fix:
        return apply_fix(args.fix)
    
    # Diagnosis mode
    gpu_info = check_gpu_memory()
    oom_info = analyze_log_for_oom()
    failed_count = count_failed_experiments()
    
    print_diagnosis(gpu_info, oom_info, failed_count)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
