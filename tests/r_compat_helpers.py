"""Shared helpers for R-compatibility tests.

Importable by any test module.  Pytest fixtures live in conftest.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixtures_present(*filenames: str) -> bool:
    return all((FIXTURES_DIR / f).exists() for f in filenames)


def load_fixture_vector(name: str) -> np.ndarray:
    """Load a single-column CSV fixture as a 1-D numpy array."""
    df = pd.read_csv(FIXTURES_DIR / name)
    return df.iloc[:, 0].to_numpy(dtype=float)


def load_fixture_matrix(name: str) -> np.ndarray:
    """Load a multi-column CSV fixture as a 2-D numpy array."""
    return pd.read_csv(FIXTURES_DIR / name).to_numpy(dtype=float)


def require_fixtures(*filenames):
    """Return a pytest.mark.skipif that skips when any fixture file is absent."""
    missing = [f for f in filenames if not (FIXTURES_DIR / f).exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"R fixture(s) not found: {missing}. Run tests/fixtures/generate_fixtures.R",
    )
