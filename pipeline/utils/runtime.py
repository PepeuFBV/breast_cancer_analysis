from __future__ import annotations


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}min")
    if secs or not parts:
        parts.append(f"{secs}sec")
    return " ".join(parts)


def resolve_bool_flag(value: bool | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value
