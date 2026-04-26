from __future__ import annotations

import pytest

KNOWN_MARKERS = ("unit", "smoke", "integration", "gpu", "slow")


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        if any(item.iter_markers(marker) for marker in KNOWN_MARKERS):
            continue
        item.add_marker(pytest.mark.unit)
