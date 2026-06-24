from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import shutil

import pytest


@pytest.fixture()
def workspace_tmp() -> Iterator[Path]:
    """Workspace-local temp directory.

    The execution sandbox used for these tests may not allow writes to the
    platform temp directory, so pytest tests use a temporary folder under the
    repository root and remove it after each test.
    """
    root = Path.cwd().resolve()
    path = root / "test_pytest_artifacts"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    try:
        yield path
    finally:
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"Refusing to remove path outside workspace: {resolved}")
        if path.exists():
            shutil.rmtree(path)
