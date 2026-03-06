"""R compatibility tests for dlnm knot utilities.

logknots and equalknots are pure arithmetic functions — they must match R
exactly (not just approximately), so we use a tighter tolerance here.
"""

import numpy as np
from r_compat_helpers import load_fixture_vector, require_fixtures

import dlnm

RTOL = 1e-10
ATOL = 1e-12


class TestLogknots:
    @require_fixtures("logknots_lag10_nk3.csv")
    def test_lag10_nk3(self):
        r_knots = load_fixture_vector("logknots_lag10_nk3.csv")
        py_knots = dlnm.logknots(10, nk=3)
        np.testing.assert_allclose(
            py_knots, r_knots, rtol=RTOL, atol=ATOL, err_msg="logknots(10, nk=3) diverges from R"
        )

    @require_fixtures("logknots_lag21_nk4.csv")
    def test_lag21_nk4(self):
        r_knots = load_fixture_vector("logknots_lag21_nk4.csv")
        py_knots = dlnm.logknots(21, nk=4)
        np.testing.assert_allclose(
            py_knots, r_knots, rtol=RTOL, atol=ATOL, err_msg="logknots(21, nk=4) diverges from R"
        )

    # -----------------------------------------------------------------------
    # Structural invariants (no fixture needed)
    # -----------------------------------------------------------------------

    def test_returns_nk_knots(self):
        for nk in (1, 2, 3, 5):
            assert len(dlnm.logknots(21, nk=nk)) == nk

    def test_knots_within_range(self):
        knots = dlnm.logknots(21, nk=4)
        assert np.all(knots > 0)
        assert np.all(knots < 21)

    def test_knots_strictly_increasing(self):
        knots = dlnm.logknots(21, nk=4)
        assert np.all(np.diff(knots) > 0)

    def test_lag_range_tuple(self):
        """logknots([0, 21], nk=3) should behave like logknots(21, nk=3)."""
        k1 = dlnm.logknots(21, nk=3)
        k2 = dlnm.logknots([0, 21], nk=3)
        np.testing.assert_array_equal(k1, k2)


class TestEqualknots:
    @require_fixtures("equalknots_lag30_nk3.csv")
    def test_lag30_nk3(self):
        r_knots = load_fixture_vector("equalknots_lag30_nk3.csv")
        py_knots = dlnm.equalknots(np.arange(0, 31, dtype=float), nk=3)
        np.testing.assert_allclose(
            py_knots,
            r_knots,
            rtol=RTOL,
            atol=ATOL,
            err_msg="equalknots(0:30, nk=3) diverges from R",
        )

    @require_fixtures("equalknots_lag30_nk4.csv")
    def test_lag30_nk4(self):
        r_knots = load_fixture_vector("equalknots_lag30_nk4.csv")
        py_knots = dlnm.equalknots(np.arange(0, 31, dtype=float), nk=4)
        np.testing.assert_allclose(
            py_knots,
            r_knots,
            rtol=RTOL,
            atol=ATOL,
            err_msg="equalknots(0:30, nk=4) diverges from R",
        )

    def test_returns_nk_knots(self):
        x = np.arange(0, 31, dtype=float)
        for nk in (1, 2, 3, 5):
            assert len(dlnm.equalknots(x, nk=nk)) == nk

    def test_knots_equally_spaced(self):
        x = np.arange(0, 31, dtype=float)
        knots = dlnm.equalknots(x, nk=4)
        diffs = np.diff(knots)
        np.testing.assert_allclose(
            diffs,
            diffs[0] * np.ones_like(diffs),
            rtol=1e-10,
            err_msg="equalknots not equally spaced",
        )

    def test_knots_within_range(self):
        x = np.arange(0, 31, dtype=float)
        knots = dlnm.equalknots(x, nk=4)
        assert np.all(knots > 0)
        assert np.all(knots < 30)
