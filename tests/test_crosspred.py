"""Tests for pydlnm.crosspred module."""

import numpy as np
import pytest

from pydlnm.crossbasis import crossbasis
from pydlnm.crosspred import CrossPred, crosspred


@pytest.fixture
def simple_fit():
    """Create a simple cross-basis and fake coefficients for testing."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    cb = crossbasis(
        x,
        lag=5,
        argvar={"fun": "ns", "df": 3},
        arglag={"fun": "ns", "df": 2},
    )
    n_params = cb.shape[1]
    coef = np.random.randn(n_params) * 0.01
    vcov = np.eye(n_params) * 0.001
    return cb, coef, vcov


class TestCrosspred:
    def test_basic(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred = crosspred(cb, coef=coef, vcov=vcov)
        assert isinstance(pred, CrossPred)
        assert len(pred.predvar) > 0
        assert pred.matfit.shape[0] == len(pred.predvar)
        assert pred.matfit.shape[1] == 6  # lag 0..5
        assert len(pred.allfit) == len(pred.predvar)

    def test_with_at(self, simple_fit):
        cb, coef, vcov = simple_fit
        at = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        pred = crosspred(cb, coef=coef, vcov=vcov, at=at)
        assert len(pred.predvar) == 5
        assert pred.matfit.shape[0] == 5

    def test_cumulative(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred = crosspred(cb, coef=coef, vcov=vcov, cumul=True)
        assert pred.cumfit is not None
        assert pred.cumse is not None
        assert pred.cumfit.shape == pred.matfit.shape

    def test_ci_bounds_linear(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred = crosspred(cb, coef=coef, vcov=vcov)
        assert pred.matlow is not None
        assert pred.mathigh is not None
        assert pred.alllow is not None
        assert pred.allhigh is not None
        # Low should be below fit, high above
        assert (pred.matlow <= pred.matfit + 1e-10).all()
        assert (pred.mathigh >= pred.matfit - 1e-10).all()

    def test_ci_bounds_log_link(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred = crosspred(cb, coef=coef, vcov=vcov, model_link="log")
        assert pred.matRRfit is not None
        assert pred.matRRlow is not None
        assert pred.matRRhigh is not None
        assert pred.allRRfit is not None
        # Exponentiated values should be positive
        assert (pred.matRRfit > 0).all()
        assert (pred.allRRfit > 0).all()

    def test_centering(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred1 = crosspred(cb, coef=coef, vcov=vcov, cen=20.0)
        assert pred1.cen == 20.0
        # At the centering value, the overall effect should be small
        cen_idx = np.argmin(np.abs(pred1.predvar - 20.0))
        assert abs(pred1.allfit[cen_idx]) < 0.1

    def test_no_model_no_coef_raises(self, simple_fit):
        cb, _, _ = simple_fit
        with pytest.raises(ValueError, match="At least 'model'"):
            crosspred(cb)

    def test_wrong_coef_size_raises(self, simple_fit):
        cb, _, vcov = simple_fit
        with pytest.raises(ValueError, match="not consistent"):
            crosspred(cb, coef=np.array([1.0, 2.0]), vcov=vcov)

    def test_ci_level(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred_90 = crosspred(cb, coef=coef, vcov=vcov, ci_level=0.90)
        pred_99 = crosspred(cb, coef=coef, vcov=vcov, ci_level=0.99)
        # 99% CI should be wider than 90%
        width_90 = pred_90.mathigh - pred_90.matlow
        width_99 = pred_99.mathigh - pred_99.matlow
        assert (width_99 >= width_90 - 1e-10).all()

    def test_summary(self, simple_fit):
        cb, coef, vcov = simple_fit
        pred = crosspred(cb, coef=coef, vcov=vcov)
        s = pred.summary()
        assert "PREDICTIONS" in s
        assert "MODEL" in s


class TestCrosspredOneBasis:
    def test_onebasis_pred(self):
        from pydlnm.basis import onebasis

        np.random.seed(42)
        x = np.random.randn(100) * 5 + 20
        ob = onebasis(x, fun="ns", df=4)
        coef = np.random.randn(ob.shape[1]) * 0.01
        vcov = np.eye(ob.shape[1]) * 0.001
        pred = crosspred(ob, coef=coef, vcov=vcov)
        assert isinstance(pred, CrossPred)
        assert len(pred.predvar) > 0
