import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.integration`` tests unless ``RUN_INTEGRATION=1``.

    These tests hit real models/network (e.g. downloading the production
    embedding model) and are excluded from the default fast test run.
    """
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip_integration = pytest.mark.skip(reason="set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
