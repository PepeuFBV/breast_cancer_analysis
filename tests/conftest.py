from __future__ import annotations

import pytest

KNOWN_MARKERS = ("unit", "smoke", "integration", "gpu", "slow", "memory", "stress")


def pytest_configure(config) -> None:
    for marker in KNOWN_MARKERS:
        config.addinivalue_line("markers", f"{marker}: categorized test marker")


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        if any(item.iter_markers(marker) for marker in KNOWN_MARKERS):
            continue
        item.add_marker(pytest.mark.unit)
