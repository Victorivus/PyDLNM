"""R compatibility tests for pydlnm.crossbasis.

Compares the full crossbasis matrix (element-by-element) against golden
fixtures produced by R's dlnm::crossbasis().

Run Rscript tests/fixtures/generate_fixtures.R to regenerate fixtures.
"""

import numpy as np
import pytest
from r_compat_helpers import load_fixture_matrix, require_fixtures

import pydlnm

RTOL = 1e-5
ATOL = 1e-9


def _check_cb(py_cb: np.ndarray, r_csv: str) -> None:
    r_mat = load_fixture_matrix(r_csv)
    np.testing.assert_allclose(
        py_cb,
        r_mat,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"crossbasis mismatch vs R fixture '{r_csv}'",
    )


class TestCrossbasisR:
    @require_fixtures("synthetic_x.csv", "crossbasis_ns_df4_ns_df3_lag10.csv")
    def test_ns_ns_lag10(self, synthetic_x):
        """Primary case: ns(df=4) x ns(df=3), lag=10 → 12 columns."""
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape == (len(synthetic_x), 12)
        _check_cb(np.asarray(cb), "crossbasis_ns_df4_ns_df3_lag10.csv")

    @require_fixtures("synthetic_x.csv", "crossbasis_bs_df4_ns_df3_lag10.csv")
    def test_bs_ns_lag10(self, synthetic_x):
        """bs(df=4) x ns(df=3), lag=10 → 12 columns."""
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "bs", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape == (len(synthetic_x), 12)
        _check_cb(np.asarray(cb), "crossbasis_bs_df4_ns_df3_lag10.csv")

    @require_fixtures("synthetic_x.csv", "crossbasis_lin_ns_df3_lag10.csv")
    def test_lin_ns_lag10(self, synthetic_x):
        """lin x ns(df=3), lag=10 → 3 columns."""
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "lin"},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape == (len(synthetic_x), 3)
        _check_cb(np.asarray(cb), "crossbasis_lin_ns_df3_lag10.csv")

    @require_fixtures("synthetic_x.csv", "crossbasis_ns_df4_strata_df2_lag10.csv")
    def test_ns_strata_lag10(self, synthetic_x):
        """ns(df=4) x strata(df=2), lag=10 → 8 columns."""
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "strata", "df": 2},
        )
        assert cb.shape == (len(synthetic_x), 8)
        _check_cb(np.asarray(cb), "crossbasis_ns_df4_strata_df2_lag10.csv")

    @require_fixtures("synthetic_x.csv", "crossbasis_ns_df4_ns_df3_lag0_21.csv")
    def test_ns_ns_lag0_21(self, synthetic_x):
        """ns(df=4) x ns(df=3), lag=[0,21] → 12 columns, 22 lag periods."""
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=[0, 21],
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.shape == (len(synthetic_x), 12)
        np.testing.assert_array_equal(cb.lag, [0, 21])
        _check_cb(np.asarray(cb), "crossbasis_ns_df4_ns_df3_lag0_21.csv")

    # -----------------------------------------------------------------------
    # Structural / metadata tests (no fixture required)
    # -----------------------------------------------------------------------

    def test_attributes_preserved(self, synthetic_x):
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.argvar["fun"] == "ns"
        assert cb.arglag["fun"] == "ns"
        np.testing.assert_array_equal(cb.lag, [0, 10])
        np.testing.assert_array_equal(cb.df, [4, 3])

    def test_range_matches_data(self, synthetic_x):
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=10,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        assert cb.range_[0] == pytest.approx(np.min(synthetic_x))
        assert cb.range_[1] == pytest.approx(np.max(synthetic_x))

    def test_no_nan_after_lag_warmup(self, synthetic_x):
        lag = 10
        cb = pydlnm.crossbasis(
            synthetic_x,
            lag=lag,
            argvar={"fun": "ns", "df": 4},
            arglag={"fun": "ns", "df": 3},
        )
        # Rows beyond the lag warm-up period must be finite
        assert np.isfinite(np.asarray(cb)[lag:]).all()
