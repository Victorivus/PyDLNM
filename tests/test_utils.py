"""Tests for pydlnm.utils module."""

import numpy as np
import pytest

from pydlnm.utils import (
    exphist,
    findrank,
    lag_matrix,
    mkat,
    mkcen,
    mklag,
    seqlag,
    tensor_product,
)


class TestMklag:
    def test_single_positive(self):
        result = mklag(5)
        np.testing.assert_array_equal(result, [0, 5])

    def test_single_negative(self):
        result = mklag(-3)
        np.testing.assert_array_equal(result, [-3, 0])

    def test_pair(self):
        result = mklag([2, 10])
        np.testing.assert_array_equal(result, [2, 10])

    def test_pair_with_zero(self):
        result = mklag([0, 7])
        np.testing.assert_array_equal(result, [0, 7])

    def test_rounding(self):
        result = mklag([0.3, 5.7])
        np.testing.assert_array_equal(result, [0, 6])

    def test_reversed_raises(self):
        with pytest.raises(ValueError, match="lag\\[0\\] must be <= lag\\[1\\]"):
            mklag([10, 2])

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            mklag([1, 2, 3])


class TestSeqlag:
    def test_default(self):
        result = seqlag(np.array([0, 5]))
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4, 5])

    def test_by_2(self):
        result = seqlag(np.array([0, 6]), by=2)
        np.testing.assert_array_equal(result, [0, 2, 4, 6])


class TestMkat:
    def test_default_generates_values(self):
        result = mkat(None, None, None, None, np.array([0.0, 30.0]), np.array([0, 5]), 1)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
        assert result.min() >= 0.0
        assert result.max() <= 30.0

    def test_explicit_at(self):
        at = np.array([1.0, 5.0, 10.0])
        result = mkat(at, None, None, None, np.array([0.0, 30.0]), np.array([0, 5]), 1)
        np.testing.assert_array_equal(result, [1.0, 5.0, 10.0])

    def test_from_to_by(self):
        result = mkat(None, 0.0, 10.0, 2.0, np.array([0.0, 10.0]), np.array([0, 5]), 1)
        assert len(result) > 0


class TestLagMatrix:
    def test_simple_lags(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lag_seq = np.array([0, 1, 2])
        result = lag_matrix(x, lag_seq)
        assert result.shape == (5, 3)
        # Lag 0: same as x
        np.testing.assert_array_equal(result[:, 0], x)
        # Lag 1: shifted by 1
        assert np.isnan(result[0, 1])
        np.testing.assert_array_equal(result[1:, 1], [1.0, 2.0, 3.0, 4.0])
        # Lag 2: shifted by 2
        assert np.isnan(result[0, 2])
        assert np.isnan(result[1, 2])
        np.testing.assert_array_equal(result[2:, 2], [1.0, 2.0, 3.0])


class TestTensorProduct:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = tensor_product(A, B)
        assert result.shape == (2, 4)
        # Row 0: kron([1,2], [5,6]) = [5, 6, 10, 12]
        np.testing.assert_array_equal(result[0], [5, 6, 10, 12])
        # Row 1: kron([3,4], [7,8]) = [21, 24, 28, 32]
        np.testing.assert_array_equal(result[1], [21, 24, 28, 32])


class TestExphist:
    def test_basic(self):
        exp = np.array([10, 20, 30, 40, 50], dtype=float)
        result = exphist(exp, lag=2)
        assert result.shape[0] == 5
        assert result.shape[1] == 3  # lag 0, 1, 2

    def test_custom_times(self):
        exp = np.arange(1.0, 11.0)
        result = exphist(exp, times=np.array([5, 8]))
        assert result.shape[0] == 2


class TestFindrank:
    def test_full_rank(self):
        X = np.eye(5)
        assert findrank(X) == 5

    def test_rank_deficient(self):
        X = np.array([[1, 1], [1, 1]], dtype=float)
        assert findrank(X) == 1


class TestMkcen:
    def test_returns_none_when_false(self):
        result = mkcen(False, "cb", None, np.array([0.0, 30.0]))
        assert result is None

    def test_returns_value_when_numeric(self):
        result = mkcen(20.0, "cb", None, np.array([0.0, 30.0]))
        assert result == 20.0
