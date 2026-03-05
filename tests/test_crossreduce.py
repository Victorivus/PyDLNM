"""Tests for pydlnm.crossreduce module."""

import numpy as np
import pytest

from pydlnm.crossbasis import crossbasis
from pydlnm.crossreduce import CrossReduce, crossreduce


@pytest.fixture
def cb_with_coef():
    """Cross-basis and coefficients for testing."""
    np.random.seed(42)
    x = np.random.randn(200) * 5 + 20
    cb = crossbasis(
        x, lag=10,
        argvar={"fun": "ns", "df": 3},
        arglag={"fun": "ns", "df": 3},
    )
    n_params = cb.shape[1]
    coef = np.random.randn(n_params) * 0.01
    vcov = np.eye(n_params) * 0.001
    return cb, coef, vcov


class TestCrossreduce:
    def test_overall(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="overall")
        assert isinstance(red, CrossReduce)
        assert red.type == "overall"
        assert len(red.fit) > 0
        assert len(red.se) > 0
        assert red.predvar is not None

    def test_lag_specific(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="lag", value=5)
        assert red.type == "lag"
        assert red.value == 5
        assert len(red.fit) > 0

    def test_var_specific(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="var", value=20.0)
        assert red.type == "var"
        assert red.value == 20.0
        assert len(red.fit) > 0

    def test_ci_bounds_identity(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="overall")
        assert red.low is not None
        assert red.high is not None
        assert (red.low <= red.fit + 1e-10).all()
        assert (red.high >= red.fit - 1e-10).all()

    def test_ci_bounds_log(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="overall", model_link="log")
        assert red.RRfit is not None
        assert (red.RRfit > 0).all()

    def test_missing_value_raises(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        with pytest.raises(ValueError, match="'value' must be provided"):
            crossreduce(cb, coef=coef, vcov=vcov, type="lag")

    def test_not_crossbasis_raises(self):
        with pytest.raises(TypeError, match="CrossBasis"):
            crossreduce(np.eye(5), coef=np.ones(5), vcov=np.eye(5))

    def test_lag_out_of_range_raises(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        with pytest.raises(ValueError, match="within the lag range"):
            crossreduce(cb, coef=coef, vcov=vcov, type="lag", value=99)

    def test_reduced_coef_dimension(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red_overall = crossreduce(cb, coef=coef, vcov=vcov, type="overall")
        # Overall reduces lag dimension: result should have df_var coefficients
        assert len(red_overall.coefficients) == int(cb.df[0])

        red_var = crossreduce(cb, coef=coef, vcov=vcov, type="var", value=20.0)
        # Var reduces var dimension: result should have df_lag coefficients
        assert len(red_var.coefficients) == int(cb.df[1])

    def test_summary(self, cb_with_coef):
        cb, coef, vcov = cb_with_coef
        red = crossreduce(cb, coef=coef, vcov=vcov, type="overall")
        s = red.summary()
        assert "REDUCED FIT" in s
