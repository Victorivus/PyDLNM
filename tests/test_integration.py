"""Integration tests for end-to-end DLNM workflows.

These tests mimic typical usage patterns from the R dlnm package examples.
"""

import numpy as np

import pydlnm


class TestTimeSeriesWorkflow:
    """Test a typical time-series DLNM analysis."""

    def test_full_workflow_with_manual_coef(self):
        """Full pipeline: crossbasis -> crosspred -> crossreduce."""
        # Load data
        df = pydlnm.load_chicagoNMMAPS()
        temp = df["temp"].values

        # Create cross-basis
        cb = pydlnm.crossbasis(
            temp,
            lag=21,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape[0] == len(temp)
        assert cb.shape[1] == 12  # 4 * 3

        # Create fake coefficients (in real usage, from a fitted model)
        np.random.seed(123)
        coef = np.random.randn(12) * 0.001
        vcov = np.eye(12) * 0.0001

        # Cross-predictions
        pred = pydlnm.crosspred(
            cb,
            coef=coef,
            vcov=vcov,
            at=np.arange(-20, 35, dtype=float),
            cen=21.0,
        )
        assert pred.cen == 21.0
        assert pred.matfit.shape[0] == 55  # -20 to 34
        assert pred.matfit.shape[1] == 22  # lag 0..21

        # At centering value, the overall effect should be near zero
        cen_idx = np.argmin(np.abs(pred.predvar - 21.0))
        assert abs(pred.allfit[cen_idx]) < 0.1

        # Cross-reduction: overall
        red = pydlnm.crossreduce(
            cb,
            coef=coef,
            vcov=vcov,
            type="overall",
            cen=21.0,
        )
        assert red.type == "overall"
        assert len(red.fit) > 0

        # Cross-reduction: var-specific
        red_var = pydlnm.crossreduce(
            cb,
            coef=coef,
            vcov=vcov,
            type="var",
            value=30.0,
            cen=21.0,
        )
        assert red_var.type == "var"
        assert red_var.value == 30.0

    def test_logknots_workflow(self):
        """Use logknots for lag basis specification."""
        df = pydlnm.load_chicagoNMMAPS()
        temp = df["temp"].values

        knots = pydlnm.logknots(21, nk=3)
        assert len(knots) == 3

        cb = pydlnm.crossbasis(
            temp,
            lag=21,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "knots": knots},
        )
        assert cb.shape[0] == len(temp)


class TestDrugWorkflow:
    """Test DLNM with pre-lagged matrix input (drug trial data)."""

    def test_matrix_input(self):
        df = pydlnm.load_drug()
        # Build exposure history matrix
        dose_cols = ["day1.7", "day8.14", "day15.21", "day22.28"]
        # Expand weekly doses to daily (4 weeks * 7 days = 28 days)
        exp_mat = np.column_stack(
            [np.repeat(df[col].values.reshape(-1, 1), 7, axis=1) for col in dose_cols]
        )
        assert exp_mat.shape == (200, 28)

        cb = pydlnm.crossbasis(
            exp_mat,
            lag=[0, 27],
            argvar={"fun": "lin"},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape[0] == 200


class TestExphist:
    """Test exposure history construction."""

    def test_basic_exphist(self):
        exp = np.arange(1, 101, dtype=float)
        hist = pydlnm.exphist(exp, times=np.array([50, 60, 70]), lag=10)
        assert hist.shape == (3, 11)  # lag 0..10 = 11 columns


class TestEdgeCases:
    def test_zero_lag(self):
        """Zero lag should give a simple 1-D basis."""
        np.random.seed(42)
        x = np.random.randn(100)
        cb = pydlnm.crossbasis(x, lag=0, argvar={"fun": "ns", "df": 3})
        assert cb.shape[0] == 100
        assert cb.shape[1] == 3  # df_var * 1

    def test_single_lag(self):
        np.random.seed(42)
        x = np.random.randn(100)
        cb = pydlnm.crossbasis(x, lag=1, argvar={"fun": "lin"})
        assert cb.shape[0] == 100
