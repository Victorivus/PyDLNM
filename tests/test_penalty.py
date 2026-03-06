"""Tests for dlnm.penalty module."""

import numpy as np
import pytest

from dlnm.crossbasis import crossbasis
from dlnm.penalty import cb_pen


@pytest.fixture
def cb_ps():
    """Cross-basis with P-spline bases (penalised)."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    return crossbasis(
        x,
        lag=10,
        argvar={"fun": "ps", "df": 5},
        arglag={"fun": "ps", "df": 4},
    )


@pytest.fixture
def cb_ns():
    """Cross-basis with natural spline bases (unpenalised)."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    return crossbasis(
        x,
        lag=10,
        argvar={"fun": "ns", "df": 4},
        arglag={"fun": "ns", "df": 3},
    )


class TestCbPen:
    def test_ps_penalty(self, cb_ps):
        pen = cb_pen(cb_ps)
        assert "Svar" in pen
        assert "Slag" in pen
        assert "rank" in pen
        assert "sp" in pen

    def test_penalty_symmetric(self, cb_ps):
        pen = cb_pen(cb_ps)
        for key in ("Svar", "Slag"):
            if key in pen and isinstance(pen[key], np.ndarray):
                np.testing.assert_array_almost_equal(pen[key], pen[key].T)

    def test_no_penalty_for_ns_raises(self, cb_ns):
        """NS bases are not penalised, so cb_pen should raise."""
        with pytest.raises(ValueError, match="no penalisation"):
            cb_pen(cb_ns)

    def test_custom_sp(self, cb_ps):
        pen = cb_pen(cb_ps, sp=np.array([0.1, 0.2]))
        np.testing.assert_array_equal(pen["sp"], [0.1, 0.2])
