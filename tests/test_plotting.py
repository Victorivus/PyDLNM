"""Tests for pydlnm.plotting module."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing

import numpy as np
import pytest

from pydlnm.crossbasis import crossbasis
from pydlnm.crosspred import crosspred
from pydlnm.crossreduce import crossreduce
from pydlnm.plotting import plot_crosspred, plot_crossreduce


@pytest.fixture
def pred():
    """Create a CrossPred for plotting tests."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    cb = crossbasis(
        x, lag=10,
        argvar={"fun": "ns", "df": 3},
        arglag={"fun": "ns", "df": 3},
    )
    n = cb.shape[1]
    coef = np.random.randn(n) * 0.01
    vcov = np.eye(n) * 0.001
    return crosspred(cb, coef=coef, vcov=vcov, cumul=True)


@pytest.fixture
def red():
    """Create a CrossReduce for plotting tests."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    cb = crossbasis(
        x, lag=10,
        argvar={"fun": "ns", "df": 3},
        arglag={"fun": "ns", "df": 3},
    )
    n = cb.shape[1]
    coef = np.random.randn(n) * 0.01
    vcov = np.eye(n) * 0.001
    return crossreduce(cb, coef=coef, vcov=vcov, type="overall")


class TestPlotCrosspred:
    def test_3d(self, pred):
        fig = plot_crosspred(pred, ptype="3d")
        assert fig is not None

    def test_contour(self, pred):
        fig = plot_crosspred(pred, ptype="contour")
        assert fig is not None

    def test_overall(self, pred):
        fig = plot_crosspred(pred, ptype="overall")
        assert fig is not None

    def test_slices_lag(self, pred):
        lag_val = 5
        fig = plot_crosspred(pred, ptype="slices", lag=lag_val)
        assert fig is not None

    def test_slices_var(self, pred):
        var_val = pred.predvar[len(pred.predvar) // 2]
        fig = plot_crosspred(pred, ptype="slices", var=var_val)
        assert fig is not None

    def test_cumulative(self, pred):
        fig = plot_crosspred(pred, ptype="overall", cumul=False)
        assert fig is not None

    def test_ci_styles(self, pred):
        for ci_style in ["area", "bars", "lines", "n"]:
            fig = plot_crosspred(pred, ptype="overall", ci=ci_style)
            assert fig is not None


class TestPlotCrossreduce:
    def test_overall(self, red):
        fig = plot_crossreduce(red)
        assert fig is not None

    def test_ci_styles(self, red):
        for ci_style in ["area", "lines", "n"]:
            fig = plot_crossreduce(red, ci=ci_style)
            assert fig is not None
