"""R compatibility tests for pydlnm.crossreduce.

Uses the same crossbasis (ns/ns, lag=10) and fixed coefficients as the
crosspred tests.  Three reduction types are tested:

  - overall  : cumulative effect over all lags, at each predictor value
  - var      : lag-response at a fixed exposure value (x=30)
  - lag      : exposure-response at a fixed lag (lag=5)
"""

import numpy as np
import pytest

import pydlnm
from r_compat_helpers import (
    load_fixture_vector,
    require_fixtures,
)

RTOL = 1e-5
ATOL = 1e-9

PRED_AT = np.arange(-10, 36, dtype=float)
CEN_VAL = 20.0


@pytest.fixture(scope="module")
def cb_ns_ns(synthetic_x):
    return pydlnm.crossbasis(
        synthetic_x,
        lag=10,
        argvar={"fun": "ns", "df": 4},
        arglag={"fun": "ns", "df": 3},
    )


@pytest.fixture(scope="module")
def red_overall(cb_ns_ns, coef_12, vcov_12):
    return pydlnm.crossreduce(
        cb_ns_ns,
        coef=coef_12,
        vcov=vcov_12,
        type="overall",
        at=PRED_AT,
        cen=CEN_VAL,
    )


@pytest.fixture(scope="module")
def red_var(cb_ns_ns, coef_12, vcov_12):
    return pydlnm.crossreduce(
        cb_ns_ns,
        coef=coef_12,
        vcov=vcov_12,
        type="var",
        value=30.0,
        cen=CEN_VAL,
    )


@pytest.fixture(scope="module")
def red_lag(cb_ns_ns, coef_12, vcov_12):
    return pydlnm.crossreduce(
        cb_ns_ns,
        coef=coef_12,
        vcov=vcov_12,
        type="lag",
        value=5.0,
        at=PRED_AT,
        cen=CEN_VAL,
    )


# ---------------------------------------------------------------------------
# Overall reduction
# ---------------------------------------------------------------------------


class TestCrossreduceOverall:
    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_overall_fit.csv")
    def test_fit(self, red_overall):
        r = load_fixture_vector("crossreduce_overall_fit.csv")
        np.testing.assert_allclose(red_overall.fit, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce overall fit diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_overall_se.csv")
    def test_se(self, red_overall):
        r = load_fixture_vector("crossreduce_overall_se.csv")
        np.testing.assert_allclose(red_overall.se, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce overall se diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_overall_low.csv")
    def test_low(self, red_overall):
        r = load_fixture_vector("crossreduce_overall_low.csv")
        np.testing.assert_allclose(red_overall.low, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce overall low diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_overall_high.csv")
    def test_high(self, red_overall):
        r = load_fixture_vector("crossreduce_overall_high.csv")
        np.testing.assert_allclose(red_overall.high, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce overall high diverges from R")

    # Structural
    def test_type(self, red_overall):
        assert red_overall.type == "overall"

    def test_se_non_negative(self, red_overall):
        assert np.all(red_overall.se >= 0)

    def test_ci_contains_fit(self, red_overall):
        assert np.all(red_overall.low <= red_overall.fit + 1e-12)
        assert np.all(red_overall.fit <= red_overall.high + 1e-12)

    def test_fit_zero_at_cen(self, red_overall):
        """At centering value the reduced effect must be zero."""
        cen_idx = np.argmin(np.abs(red_overall.predvar - CEN_VAL))
        assert abs(red_overall.fit[cen_idx]) < 1e-10


# ---------------------------------------------------------------------------
# Var-specific reduction (lag-response at x=30)
# ---------------------------------------------------------------------------


class TestCrossreduceVar:
    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_var_fit.csv")
    def test_fit(self, red_var):
        r = load_fixture_vector("crossreduce_var_fit.csv")
        np.testing.assert_allclose(red_var.fit, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce var fit diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_var_se.csv")
    def test_se(self, red_var):
        r = load_fixture_vector("crossreduce_var_se.csv")
        np.testing.assert_allclose(red_var.se, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce var se diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_var_low.csv")
    def test_low(self, red_var):
        r = load_fixture_vector("crossreduce_var_low.csv")
        np.testing.assert_allclose(red_var.low, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce var low diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_var_high.csv")
    def test_high(self, red_var):
        r = load_fixture_vector("crossreduce_var_high.csv")
        np.testing.assert_allclose(red_var.high, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce var high diverges from R")

    def test_type_and_value(self, red_var):
        assert red_var.type == "var"
        assert red_var.value == pytest.approx(30.0)

    def test_len_equals_lag_period(self, red_var):
        """Var reduction produces one value per lag (0..10 = 11 values)."""
        assert len(red_var.fit) == 11


# ---------------------------------------------------------------------------
# Lag-specific reduction (exposure-response at lag=5)
# ---------------------------------------------------------------------------


class TestCrossreduceLag:
    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_lag_fit.csv")
    def test_fit(self, red_lag):
        r = load_fixture_vector("crossreduce_lag_fit.csv")
        np.testing.assert_allclose(red_lag.fit, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce lag fit diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_lag_se.csv")
    def test_se(self, red_lag):
        r = load_fixture_vector("crossreduce_lag_se.csv")
        np.testing.assert_allclose(red_lag.se, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce lag se diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_lag_low.csv")
    def test_low(self, red_lag):
        r = load_fixture_vector("crossreduce_lag_low.csv")
        np.testing.assert_allclose(red_lag.low, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce lag low diverges from R")

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crossreduce_lag_high.csv")
    def test_high(self, red_lag):
        r = load_fixture_vector("crossreduce_lag_high.csv")
        np.testing.assert_allclose(red_lag.high, r, rtol=RTOL, atol=ATOL,
                                   err_msg="crossreduce lag high diverges from R")

    def test_type_and_value(self, red_lag):
        assert red_lag.type == "lag"
        assert red_lag.value == pytest.approx(5.0)

    def test_len_equals_at(self, red_lag):
        assert len(red_lag.fit) == len(PRED_AT)

    def test_fit_zero_at_cen(self, red_lag):
        cen_idx = np.argmin(np.abs(red_lag.predvar - CEN_VAL))
        assert abs(red_lag.fit[cen_idx]) < 1e-10
