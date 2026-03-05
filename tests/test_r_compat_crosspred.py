"""R compatibility tests for pydlnm.crosspred.

The reference crossbasis is ns(df=4) x ns(df=3) with lag=10, applied to the
200-element synthetic_x vector.  Predictions use:
  - Fixed coef_12 from coef_12.csv
  - vcov = diag(12) * 0.0001
  - at = seq(-10, 35, by=1)  (46 values)
  - cen = 20

All fixtures are produced by R dlnm::crosspred().
"""

import numpy as np
import pytest
from r_compat_helpers import (
    load_fixture_matrix,
    load_fixture_vector,
    require_fixtures,
)

import pydlnm

RTOL = 1e-5
ATOL = 1e-9

# Prediction grid and centering (must match generate_fixtures.R)
PRED_AT = np.arange(-10, 36, dtype=float)  # -10, -9, ..., 35  (46 values)
CEN_VAL = 20.0


@pytest.fixture(scope="module")
def cb_ns_ns(synthetic_x):
    """Shared crossbasis for all crosspred tests."""
    return pydlnm.crossbasis(
        synthetic_x,
        lag=10,
        argvar={"fun": "ns", "df": 4},
        arglag={"fun": "ns", "df": 3},
    )


@pytest.fixture(scope="module")
def pred(cb_ns_ns, coef_12, vcov_12):
    """Shared crosspred result."""
    return pydlnm.crosspred(
        cb_ns_ns,
        coef=coef_12,
        vcov=vcov_12,
        at=PRED_AT,
        cen=CEN_VAL,
    )


# ---------------------------------------------------------------------------
# Overall (all-lag-summed) effects
# ---------------------------------------------------------------------------


class TestCrosspredOverall:
    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_allfit.csv")
    def test_allfit(self, pred):
        r = load_fixture_vector("crosspred_allfit.csv")
        np.testing.assert_allclose(
            pred.allfit, r, rtol=RTOL, atol=ATOL, err_msg="crosspred allfit diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_allse.csv")
    def test_allse(self, pred):
        r = load_fixture_vector("crosspred_allse.csv")
        np.testing.assert_allclose(
            pred.allse, r, rtol=RTOL, atol=ATOL, err_msg="crosspred allse diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_alllow.csv")
    def test_alllow(self, pred):
        r = load_fixture_vector("crosspred_alllow.csv")
        np.testing.assert_allclose(
            pred.alllow, r, rtol=RTOL, atol=ATOL, err_msg="crosspred alllow diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_allhigh.csv")
    def test_allhigh(self, pred):
        r = load_fixture_vector("crosspred_allhigh.csv")
        np.testing.assert_allclose(
            pred.allhigh, r, rtol=RTOL, atol=ATOL, err_msg="crosspred allhigh diverges from R"
        )


# ---------------------------------------------------------------------------
# Lag-specific matrix effects
# ---------------------------------------------------------------------------


class TestCrosspredMatrix:
    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_matfit.csv")
    def test_matfit(self, pred):
        r = load_fixture_matrix("crosspred_matfit.csv")
        np.testing.assert_allclose(
            pred.matfit, r, rtol=RTOL, atol=ATOL, err_msg="crosspred matfit diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_matse.csv")
    def test_matse(self, pred):
        r = load_fixture_matrix("crosspred_matse.csv")
        np.testing.assert_allclose(
            pred.matse, r, rtol=RTOL, atol=ATOL, err_msg="crosspred matse diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_matlow.csv")
    def test_matlow(self, pred):
        r = load_fixture_matrix("crosspred_matlow.csv")
        np.testing.assert_allclose(
            pred.matlow, r, rtol=RTOL, atol=ATOL, err_msg="crosspred matlow diverges from R"
        )

    @require_fixtures("synthetic_x.csv", "coef_12.csv", "crosspred_mathigh.csv")
    def test_mathigh(self, pred):
        r = load_fixture_matrix("crosspred_mathigh.csv")
        np.testing.assert_allclose(
            pred.mathigh, r, rtol=RTOL, atol=ATOL, err_msg="crosspred mathigh diverges from R"
        )


# ---------------------------------------------------------------------------
# Structural invariants (no fixture required)
# ---------------------------------------------------------------------------


class TestCrosspredStructure:
    def test_predvar_matches_at(self, pred):
        np.testing.assert_array_equal(pred.predvar, PRED_AT)

    def test_cen_stored(self, pred):
        assert pred.cen == pytest.approx(CEN_VAL)

    def test_shapes(self, pred):
        n_at = len(PRED_AT)
        n_lag = 11  # lag 0..10
        assert pred.allfit.shape == (n_at,)
        assert pred.allse.shape == (n_at,)
        assert pred.matfit.shape == (n_at, n_lag)
        assert pred.matse.shape == (n_at, n_lag)

    def test_allse_non_negative(self, pred):
        assert np.all(pred.allse >= 0)

    def test_matse_non_negative(self, pred):
        assert np.all(pred.matse >= 0)

    def test_ci_contains_fit(self, pred):
        """alllow <= allfit <= allhigh (95 % CI)."""
        assert np.all(pred.alllow <= pred.allfit + 1e-12)
        assert np.all(pred.allfit <= pred.allhigh + 1e-12)

    def test_allfit_zero_at_cen(self, pred):
        """At the centering value, the overall effect must be zero."""
        cen_idx = np.argmin(np.abs(pred.predvar - CEN_VAL))
        assert abs(pred.allfit[cen_idx]) < 1e-10
