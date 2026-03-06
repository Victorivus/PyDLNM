"""Tests for dlnm.basis module."""

import numpy as np
import pytest

from dlnm.basis import (
    OneBasis,
    bs,
    cr,
    integer,
    lin,
    ns,
    onebasis,
    poly,
    ps,
    strata,
    thr,
)


@pytest.fixture
def x_data():
    """Generate a sample predictor vector."""
    np.random.seed(42)
    return np.random.randn(100) * 5 + 20


@pytest.fixture
def x_int():
    """Integer sequence for lag-like data."""
    return np.arange(0, 21, dtype=float)


class TestOneBasis:
    def test_dispatch_ns(self, x_data):
        result = onebasis(x_data, fun="ns", df=4)
        assert isinstance(result, OneBasis)
        assert result.shape[0] == len(x_data)
        assert result.fun == "ns"

    def test_dispatch_bs(self, x_data):
        result = onebasis(x_data, fun="bs", df=5)
        assert isinstance(result, OneBasis)
        assert result.shape[0] == len(x_data)
        assert result.fun == "bs"

    def test_dispatch_lin(self, x_data):
        result = onebasis(x_data, fun="lin")
        assert isinstance(result, OneBasis)
        assert result.shape == (len(x_data), 1)
        assert result.fun == "lin"

    def test_dispatch_unknown_raises(self, x_data):
        with pytest.raises(ValueError, match="Unknown basis function"):
            onebasis(x_data, fun="nonexistent")


class TestNs:
    def test_shape_with_df(self, x_data):
        result = ns(x_data, df=4)
        assert result.shape == (100, 4)

    def test_shape_with_knots(self, x_data):
        knots = np.array([15.0, 20.0, 25.0])
        result = ns(x_data, knots=knots)
        assert result.shape[0] == 100
        assert result.shape[1] > 0

    def test_with_intercept(self, x_data):
        r1 = ns(x_data, df=4, intercept=False)
        r2 = ns(x_data, df=4, intercept=True)
        # With intercept should have more columns
        assert r2.shape[1] >= r1.shape[1]

    def test_handles_nan(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = ns(x, df=3)
        assert np.isnan(result[2]).all()
        assert not np.isnan(result[0]).any()


class TestBs:
    def test_shape(self, x_data):
        result = bs(x_data, df=5)
        assert result.shape[0] == 100

    def test_degree(self, x_data):
        r1 = bs(x_data, df=5, degree=2)
        r2 = bs(x_data, df=5, degree=3)
        assert r1.shape[0] == r2.shape[0]


class TestPs:
    def test_shape(self, x_data):
        result = ps(x_data, df=10)
        assert result.shape[0] == 100
        assert result.fun == "ps"

    def test_penalty_matrix(self, x_data):
        result = ps(x_data, df=10)
        assert result.S is not None
        assert result.S.shape[0] == result.shape[1]
        assert result.S.shape[1] == result.shape[1]
        # Penalty should be symmetric
        np.testing.assert_array_almost_equal(result.S, result.S.T)

    def test_fx_no_penalty(self, x_data):
        result = ps(x_data, df=10, fx=True)
        assert result.S is None

    def test_small_df_raises(self, x_data):
        with pytest.raises(ValueError):
            ps(x_data, df=2)


class TestCr:
    def test_shape(self, x_data):
        result = cr(x_data, df=10)
        assert result.shape[0] == 100
        assert result.fun == "cr"

    def test_with_knots(self, x_data):
        knots = np.quantile(x_data, np.linspace(0, 1, 12))
        result = cr(x_data, knots=knots)
        assert result.shape[0] == 100

    def test_small_df_raises(self, x_data):
        with pytest.raises(ValueError, match="'df' must be >= 3"):
            cr(x_data, df=2)


class TestStrata:
    def test_default(self, x_int):
        result = strata(x_int, df=3)
        assert result.shape[0] == len(x_int)
        assert result.fun == "strata"

    def test_with_breaks(self, x_int):
        result = strata(x_int, breaks=np.array([5.0, 10.0, 15.0]))
        assert result.shape[0] == len(x_int)

    def test_intercept(self, x_int):
        result = strata(x_int, df=3, intercept=True)
        assert result.shape[0] == len(x_int)

    def test_indicator_sums_to_one(self, x_int):
        result = strata(x_int, df=3, intercept=False)
        # Each row should have at most one 1
        row_sums = np.sum(result, axis=1)
        assert all(s <= 1 for s in row_sums)


class TestThr:
    def test_high_side(self, x_data):
        result = thr(x_data, thr_value=20.0, side="h")
        assert result.shape == (100, 1)
        assert (result[x_data <= 20.0] == 0).all()
        assert (result[x_data > 20.0] > 0).all()

    def test_low_side(self, x_data):
        result = thr(x_data, thr_value=20.0, side="l")
        assert result.shape == (100, 1)
        assert (result[x_data >= 20.0] == 0).all()

    def test_double_side(self, x_data):
        result = thr(x_data, thr_value=np.array([15.0, 25.0]), side="d")
        assert result.shape == (100, 2)

    def test_with_intercept(self, x_data):
        r1 = thr(x_data, thr_value=20.0, side="h", intercept=False)
        r2 = thr(x_data, thr_value=20.0, side="h", intercept=True)
        assert r2.shape[1] == r1.shape[1] + 1


class TestInteger:
    def test_basic(self, x_int):
        result = integer(x_int)
        assert result.shape[0] == len(x_int)
        assert result.fun == "integer"

    def test_with_values(self):
        x = np.array([0, 1, 2, 3, 1, 2], dtype=float)
        result = integer(x, values=np.array([0, 1, 2, 3]))
        assert result.shape[0] == 6

    def test_intercept(self):
        x = np.array([0, 1, 2], dtype=float)
        r1 = integer(x, intercept=False)
        r2 = integer(x, intercept=True)
        assert r2.shape[1] > r1.shape[1]


class TestLin:
    def test_shape(self, x_data):
        result = lin(x_data)
        assert result.shape == (100, 1)
        np.testing.assert_array_almost_equal(result[:, 0], x_data)

    def test_intercept(self, x_data):
        result = lin(x_data, intercept=True)
        assert result.shape == (100, 2)
        np.testing.assert_array_equal(result[:, 0], 1.0)


class TestPoly:
    def test_linear(self, x_data):
        result = poly(x_data, degree=1)
        assert result.shape == (100, 1)

    def test_quadratic(self, x_data):
        result = poly(x_data, degree=2)
        assert result.shape == (100, 2)

    def test_with_intercept(self, x_data):
        result = poly(x_data, degree=2, intercept=True)
        assert result.shape == (100, 3)
        # First column should be constant (x/scale)^0 = 1
        np.testing.assert_array_almost_equal(result[:, 0], 1.0)

    def test_scaling(self, x_data):
        result = poly(x_data, degree=1, scale=10.0)
        np.testing.assert_array_almost_equal(result[:, 0], x_data / 10.0)
