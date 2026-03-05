"""Tests for pydlnm.knots module."""

import numpy as np
import pytest

from pydlnm.knots import equalknots, logknots


class TestLogknots:
    def test_basic(self):
        knots = logknots(np.array([0, 30]), nk=3)
        assert len(knots) == 3
        assert all(0 < k < 30 for k in knots)

    def test_increasing(self):
        knots = logknots(np.array([0, 30]), nk=5)
        assert all(knots[i] < knots[i + 1] for i in range(len(knots) - 1))

    def test_log_spacing(self):
        """Log-spaced knots should be closer together at the start."""
        knots = logknots(np.array([0, 30]), nk=3)
        # First gap should be smaller than last gap
        gaps = np.diff(knots)
        assert gaps[0] < gaps[-1]

    def test_from_scalar(self):
        knots = logknots(21, nk=3)
        assert len(knots) == 3
        assert all(0 < k < 21 for k in knots)

    def test_infer_nk_from_ns(self):
        knots = logknots(30, fun="ns", df=4, intercept=True)
        assert len(knots) == 2  # df=4, intercept=True -> nk = 4-1-1 = 2

    def test_zero_range_raises(self):
        with pytest.raises(ValueError, match="range must be > 0"):
            logknots(np.array([5, 5]), nk=3)


class TestEqualknots:
    def test_basic(self):
        knots = equalknots(np.arange(0, 31, dtype=float), nk=3)
        assert len(knots) == 3

    def test_equally_spaced(self):
        knots = equalknots(np.arange(0, 31, dtype=float), nk=3)
        gaps = np.diff(knots)
        np.testing.assert_array_almost_equal(gaps, gaps[0])

    def test_within_range(self):
        x = np.arange(10, 51, dtype=float)
        knots = equalknots(x, nk=4)
        assert all(10 < k < 50 for k in knots)

    def test_infer_nk(self):
        knots = equalknots(np.arange(0, 31, dtype=float), fun="ns", df=4, intercept=False)
        assert len(knots) == 3  # df=4, intercept=False -> nk = 4-1-0 = 3
