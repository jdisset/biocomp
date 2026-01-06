"""pytest configuration for biocomp tests."""
import os
from pathlib import Path

import pytest

# Disable JAX compilation cache BEFORE importing JAX
# This is necessary because JAX's cache can cause incorrect behavior with NSGA2
# due to how closures over objects interact with JIT compilation caching.
os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "0"

# Local test resources directory (contains copies of YAML files and SVG targets)
RESOURCES_DIR = Path(__file__).parent / "resources"


@pytest.fixture(scope="module")
def test_resources_dir():
    """Return the path to the test resources directory."""
    return RESOURCES_DIR
