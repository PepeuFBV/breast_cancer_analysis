from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_for_json(val) for key, val in sorted(value.items())}
    if isinstance(value, tuple):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, list):
        return [_normalize_for_json(item) for item in value]
    return value


def _display_value(value: Any) -> str:
    if isinstance(value, dict):
        inner = ", ".join(f"{key}:{_display_value(val)}" for key, val in sorted(value.items()))
        return "{" + inner + "}"
    if isinstance(value, tuple):
        return "(" + ",".join(_display_value(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ",".join(_display_value(item) for item in value) + "]"
    return str(value)


def param_dict_to_json(params: dict[str, Any]) -> str:
    return json.dumps(_normalize_for_json(params), sort_keys=True, separators=(",", ":"))


def param_dict_to_display(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return ", ".join(f"{key}={_display_value(value)}" for key, value in sorted(params.items()))


def param_dict_to_file_id(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    digest = hashlib.sha1(param_dict_to_json(params).encode("utf-8")).hexdigest()
    return f"params-{digest[:12]}"


def _coerce_legacy_value(value: str) -> Any:
    normalized = value.strip()
    if normalized.startswith("(") and normalized.endswith(")"):
        try:
            return ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return normalized

    lowered = normalized.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in normalized:
            return float(normalized)
        return int(normalized)
    except ValueError:
        return normalized


def parse_legacy_param_combo(param_combo: str | None) -> dict[str, Any]:
    if not param_combo or param_combo == "default":
        return {}

    params: dict[str, Any] = {}
    for chunk in param_combo.split("_"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            params[key] = _coerce_legacy_value(value)
            continue
        if "-" in chunk:
            key, value = chunk.split("-", 1)
            params[key] = _coerce_legacy_value(value)
    return params
