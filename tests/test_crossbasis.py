"""Tests for pydlnm.crossbasis module."""

import numpy as np
import pytest

from pydlnm.crossbasis import CrossBasis, crossbasis


@pytest.fixture
def temp_series():
    """Simulated temperature time series."""
    np.random.seed(42)
    return np.random.randn(500) * 5 + 20


class TestCrossbasis:
    def test_basic_ns(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert isinstance(cb, CrossBasis)
        assert cb.shape[0] == 500
        assert cb.shape[1] == 4 * 3  # df_var * df_lag = 12

    def test_basic_lin(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=5,
            argvar={"fun": "lin"},
            arglag={"fun": "strata", "df": 1, "intercept": True},
        )
        assert cb.shape[0] == 500
        assert cb.shape[1] >= 1

    def test_basic_poly(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=5,
            argvar={"fun": "poly", "degree": 2},
            arglag={"fun": "poly", "degree": 1},
        )
        assert cb.shape[0] == 500

    def test_lag_range(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=[0, 21],
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        np.testing.assert_array_equal(cb.lag, [0, 21])

    def test_default_lag_basis(self, temp_series):
        """When arglag is empty, defaults to strata with df=1."""
        cb = crossbasis(
            temp_series,
            lag=0,
            argvar={"fun": "lin"},
        )
        assert cb.shape[0] == 500

    def test_matrix_input(self):
        """Test with pre-lagged matrix."""
        np.random.seed(42)
        mat = np.random.randn(100, 6)  # 6 lags
        cb = crossbasis(
            mat,
            lag=[0, 5],
            argvar={"fun": "ns", "df": 3},
            arglag={"fun": "ns", "df": 2},
        )
        assert cb.shape[0] == 100

    def test_incompatible_lag_raises(self, temp_series):
        """Matrix columns must match lag period."""
        mat = np.random.randn(100, 6)  # 6 columns = lags 0-5
        with pytest.raises(ValueError, match="NCOL"):
            crossbasis(mat, lag=[0, 10], argvar={"fun": "lin"})

    def test_attributes(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.df is not None
        assert cb.range_ is not None
        assert cb.lag is not None
        assert cb.argvar is not None
        assert cb.arglag is not None
        assert cb.argvar["fun"] == "ns"
        assert cb.arglag["fun"] == "ns"

    def test_summary(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        s = cb.summary()
        assert "CROSSBASIS" in s
        assert "observations" in s

    def test_thr_basis(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=5,
            argvar={"fun": "thr", "thr_value": 20.0, "side": "h"},
            arglag={"fun": "integer"},
        )
        assert cb.shape[0] == 500

    def test_ps_basis(self, temp_series):
        cb = crossbasis(
            temp_series,
            lag=10,
            argvar={"fun": "ps", "df": 5},
            arglag={"fun": "ps", "df": 4},
        )
        assert cb.shape[0] == 500
        assert cb.shape[1] == 5 * 4

    def test_no_nan_in_nonnan_input(self, temp_series):
        """Cross-basis should not introduce NaN for valid inputs beyond the lag period."""
        cb = crossbasis(
            temp_series,
            lag=5,
            argvar={"fun": "lin"},
            arglag={"fun": "strata", "df": 1, "intercept": True},
        )
        # After the lag period, there should be no NaN
        assert not np.isnan(cb[10:]).any()
