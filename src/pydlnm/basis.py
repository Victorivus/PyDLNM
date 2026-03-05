"""Basis function generators for DLNMs.

This module provides all the basis transformation functions used within
:func:`onebasis` and :func:`crossbasis`.  Each function takes a numeric
vector *x* and returns a design (basis) matrix along with metadata stored
as attributes on the returned :class:`OneBasis` object.

The following basis functions are available:

- :func:`ns` / :func:`bs` — natural / B-splines (via :mod:`patsy`)
- :func:`ps` — P-splines (penalised B-splines)
- :func:`cr` — natural cubic regression splines (with penalty)
- :func:`strata` — indicator (dummy) variables for strata
- :func:`thr` — threshold / hockey-stick parameterisations
- :func:`integer` — unpenalised categorical coding
- :func:`lin` — simple linear term
- :func:`poly` — scaled polynomial basis
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


# ---------------------------------------------------------------------------
# OneBasis container
# ---------------------------------------------------------------------------

class OneBasis(np.ndarray):
    """A basis matrix with metadata attributes.

    Subclasses :class:`numpy.ndarray` and carries extra information about the
    generating function, range, centering value, degrees of freedom, knots,
    and other parameters needed to reproduce the basis at prediction time.
    """

    # Attributes that travel with the array through views / slicing
    _metadata = [
        "fun", "range_", "cen", "df", "knots", "intercept",
        "degree", "Boundary_knots", "S", "fx", "diff_order",
        "thr_value", "side", "values", "scale", "breaks", "ref",
    ]

    def __new__(cls, data, **attrs):
        obj = np.asarray(data).view(cls)
        for key, val in attrs.items():
            setattr(obj, key, val)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for attr in self._metadata:
            setattr(self, attr, getattr(obj, attr, None))

    def __repr__(self):
        return f"OneBasis(shape={self.shape}, fun={self.fun!r}, df={self.df})"

    def summary(self) -> str:
        """Return a textual summary of this basis."""
        lines = [
            "BASIS FUNCTION",
            f"observations: {self.shape[0]}",
            f"range: {self.range_}",
            f"df: {self.shape[1]}",
            f"fun: {self.fun}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# onebasis — the dispatcher
# ---------------------------------------------------------------------------

_BASIS_REGISTRY: dict[str, callable] = {}


def onebasis(x: np.ndarray, fun: str = "ns", **kwargs) -> OneBasis:
    """Create a basis matrix for a predictor vector.

    This is the main entry point for constructing single-dimension basis
    transformations.  It dispatches to the function named by *fun*.

    Parameters
    ----------
    x : array-like
        Predictor values.
    fun : str, optional
        Name of the basis function (default ``"ns"``).  One of
        ``"ns"``, ``"bs"``, ``"ps"``, ``"cr"``, ``"strata"``,
        ``"thr"``, ``"integer"``, ``"lin"``, ``"poly"``.
    **kwargs
        Additional arguments forwarded to the basis function.

    Returns
    -------
    OneBasis
        Basis matrix with metadata.

    Raises
    ------
    ValueError
        If *fun* is not recognised.
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.array([np.nanmin(x), np.nanmax(x)])

    cen = kwargs.pop("cen", None)

    if fun in _BASIS_REGISTRY:
        basis = _BASIS_REGISTRY[fun](x, **kwargs)
    else:
        raise ValueError(
            f"Unknown basis function '{fun}'. Available: {list(_BASIS_REGISTRY.keys())}"
        )

    # Ensure it's an OneBasis and carry metadata
    if not isinstance(basis, OneBasis):
        basis = OneBasis(basis, fun=fun)

    basis.fun = fun
    basis.range_ = rng
    basis.cen = cen

    return basis


def _register(name: str):
    """Decorator to register a basis function."""
    def decorator(fn):
        _BASIS_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Natural splines (ns)
# ---------------------------------------------------------------------------

@_register("ns")
def ns(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[np.ndarray] = None,
    intercept: bool = False,
    Boundary_knots: Optional[np.ndarray] = None,
    **kwargs,
) -> OneBasis:
    """Natural cubic spline basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    df : int or None
        Degrees of freedom.  If *knots* is given, ``df`` is ignored.
    knots : array-like or None
        Interior knot positions.
    intercept : bool
        Whether to include an intercept column.
    Boundary_knots : array-like or None
        Boundary knot positions (default: range of *x*).

    Returns
    -------
    OneBasis
        Natural spline basis matrix.
    """
    from patsy import dmatrix
    x_clean = x.copy()
    nan_mask = np.isnan(x_clean)
    if nan_mask.any():
        x_clean[nan_mask] = 0.0  # temporary fill

    if Boundary_knots is None:
        Boundary_knots = np.array([np.nanmin(x), np.nanmax(x)])

    if knots is not None:
        knots = np.sort(np.asarray(knots, dtype=float))
        df = len(knots) + 1 + int(intercept)
    else:
        if df is None:
            df = 3
        n_inner = df - 1 - int(intercept)
        if n_inner < 0:
            n_inner = 0
        if n_inner > 0:
            probs = np.linspace(0, 1, n_inner + 2)[1:-1]
            knots = np.quantile(x_clean[~nan_mask], probs)
        else:
            knots = np.array([])

    # Build using patsy cr() which creates natural splines
    # We'll use scipy instead for better control
    basis_mat = _ns_basis(x_clean, knots, Boundary_knots, intercept)

    if nan_mask.any():
        basis_full = np.full((len(x), basis_mat.shape[1]), np.nan)
        basis_full[~nan_mask] = basis_mat[~nan_mask]
        basis_mat = basis_full

    result = OneBasis(
        basis_mat,
        fun="ns",
        df=basis_mat.shape[1],
        knots=knots,
        intercept=intercept,
        Boundary_knots=Boundary_knots,
    )
    return result


def _ns_basis(
    x: np.ndarray,
    knots: np.ndarray,
    boundary_knots: np.ndarray,
    intercept: bool,
) -> np.ndarray:
    """Compute a natural cubic spline basis using the recursive approach.

    This matches R's ``splines::ns()`` behaviour by creating a B-spline
    basis and then applying the natural spline constraint (linear beyond
    the boundary knots).
    """
    all_knots = np.sort(np.concatenate([boundary_knots, knots]))
    degree = 3

    # Augmented knot sequence for B-spline
    n_inner = len(knots)
    augmented = np.concatenate([
        np.repeat(boundary_knots[0], degree + 1),
        knots,
        np.repeat(boundary_knots[1], degree + 1),
    ])

    n_basis = len(augmented) - degree - 1  # total B-spline basis functions

    # Evaluate B-spline basis with linear extrapolation outside boundary knots.
    # R's splines::ns() (via splineDesign) extrapolates linearly beyond the
    # boundary knots.  scipy BSpline with extrapolate=True continues the cubic
    # polynomial, so we must manually apply linear extrapolation.
    basis = np.zeros((len(x), n_basis))
    left_mask = x < boundary_knots[0]
    right_mask = x > boundary_knots[1]
    interior_mask = ~left_mask & ~right_mask
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(augmented, c, degree, extrapolate=True)
        basis[:, i] = spl(x)
        if left_mask.any():
            val_l = spl(boundary_knots[0])
            slope_l = spl(boundary_knots[0], 1)
            basis[left_mask, i] = val_l + slope_l * (x[left_mask] - boundary_knots[0])
        if right_mask.any():
            val_r = spl(boundary_knots[1])
            slope_r = spl(boundary_knots[1], 1)
            basis[right_mask, i] = val_r + slope_r * (x[right_mask] - boundary_knots[1])

    # Apply natural spline constraints: project out the space that
    # allows curvature at boundaries.  The standard way is to use the
    # qr decomposition of the constraint matrix.
    # Constraint: second derivative = 0 at boundary knots
    const = np.zeros((2, n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(augmented, c, degree, extrapolate=True)
        const[0, i] = spl(boundary_knots[0], 2)  # 2nd derivative at left
        const[1, i] = spl(boundary_knots[1], 2)  # 2nd derivative at right

    # When no intercept, R excludes the first B-spline basis function by
    # adding e0 (unit vector for B-spline 0) as a third constraint row.
    # This is equivalent to R's `rbind(const, 1)` behaviour but operating
    # on the B-spline coefficient space rather than the function space.
    if not intercept:
        e0 = np.zeros((1, n_basis))
        e0[0, 0] = 1.0
        const = np.vstack([const, e0])

    # QR decomposition: null space = natural spline basis
    n_constraints = const.shape[0]  # 2 (intercept=True) or 3 (intercept=False)
    Q, _ = np.linalg.qr(const.T, mode="complete")
    ns_basis = basis @ Q[:, n_constraints:]

    return ns_basis


# ---------------------------------------------------------------------------
# B-splines (bs)
# ---------------------------------------------------------------------------

@_register("bs")
def bs(
    x: np.ndarray,
    df: Optional[int] = None,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    intercept: bool = False,
    Boundary_knots: Optional[np.ndarray] = None,
    **kwargs,
) -> OneBasis:
    """B-spline basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    df : int or None
        Degrees of freedom.
    knots : array-like or None
        Interior knot positions.
    degree : int
        Degree of the spline (default 3 = cubic).
    intercept : bool
        Whether to include an intercept column.
    Boundary_knots : array-like or None
        Boundary knots.

    Returns
    -------
    OneBasis
        B-spline basis matrix.
    """
    x_clean = x.copy()
    nan_mask = np.isnan(x_clean)
    if nan_mask.any():
        x_clean[nan_mask] = 0.0

    if Boundary_knots is None:
        Boundary_knots = np.array([np.nanmin(x), np.nanmax(x)])

    if knots is not None:
        knots = np.sort(np.asarray(knots, dtype=float))
        df = len(knots) + degree + int(intercept)
    else:
        if df is None:
            df = degree + int(intercept)
        n_inner = df - degree - int(intercept)
        if n_inner > 0:
            probs = np.linspace(0, 1, n_inner + 2)[1:-1]
            knots = np.quantile(x_clean[~nan_mask], probs)
        else:
            knots = np.array([])

    # Build augmented knot sequence
    augmented = np.concatenate([
        np.repeat(Boundary_knots[0], degree + 1),
        knots,
        np.repeat(Boundary_knots[1], degree + 1),
    ])

    n_basis = len(augmented) - degree - 1
    basis_mat = np.zeros((len(x_clean), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(augmented, c, degree, extrapolate=False)
        vals = spl(x_clean)
        vals[np.isnan(vals)] = 0.0
        # Fix boundary values
        vals[x_clean == Boundary_knots[1]] = spl(Boundary_knots[1] - 1e-10)
        basis_mat[:, i] = vals

    if not intercept:
        basis_mat = basis_mat[:, 1:]

    if nan_mask.any():
        basis_full = np.full((len(x), basis_mat.shape[1]), np.nan)
        basis_full[~nan_mask] = basis_mat[~nan_mask]
        basis_mat = basis_full

    return OneBasis(
        basis_mat,
        fun="bs",
        df=basis_mat.shape[1],
        knots=knots,
        degree=degree,
        intercept=intercept,
        Boundary_knots=Boundary_knots,
    )


# ---------------------------------------------------------------------------
# P-splines (ps)
# ---------------------------------------------------------------------------

@_register("ps")
def ps(
    x: np.ndarray,
    df: int = 10,
    knots: Optional[np.ndarray] = None,
    degree: int = 3,
    intercept: bool = False,
    fx: bool = False,
    S: Optional[np.ndarray] = None,
    diff: int = 2,
    **kwargs,
) -> OneBasis:
    """P-spline (penalised B-spline) basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    df : int
        Degrees of freedom (default 10).
    knots : array-like or None
        Full knot sequence.  If *None* or length-2 (boundary),
        inner knots are placed automatically.
    degree : int
        B-spline degree (default 3).
    intercept : bool
        Whether to include an intercept column.
    fx : bool
        If *True*, penalty matrix is not computed (fixed effects).
    S : numpy.ndarray or None
        Custom penalty matrix.
    diff : int
        Order of the difference penalty (default 2).

    Returns
    -------
    OneBasis
        P-spline basis matrix with penalty attribute ``S``.
    """
    x_clean = x.copy()
    nan_mask = np.isnan(x_clean)
    if nan_mask.any():
        x_clean[nan_mask] = 0.0

    rng = np.array([np.nanmin(x_clean[~nan_mask]), np.nanmax(x_clean[~nan_mask])])
    degree = int(degree)

    if knots is None or (isinstance(knots, np.ndarray) and len(knots) == 2):
        nik = df - degree + 2 - int(intercept)
        if nik <= 1:
            raise ValueError("basis dimension too small for b-spline degree")
        if knots is not None and len(knots) == 2:
            xl = min(knots) - np.diff(rng)[0] * 0.001
            xu = max(knots) + np.diff(rng)[0] * 0.001
        else:
            xl = rng[0] - np.diff(rng)[0] * 0.001
            xu = rng[1] + np.diff(rng)[0] * 0.001
        dx = (xu - xl) / (nik - 1)
        knots = np.linspace(xl - dx * degree, xu + dx * degree, nik + 2 * degree)
    else:
        knots = np.asarray(knots, dtype=float)
        df = len(knots) - degree - 2 + int(intercept)

    # Build B-spline basis using scipy
    n_basis = len(knots) - degree - 1
    basis_mat = np.zeros((len(x_clean), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(knots, c, degree, extrapolate=True)
        basis_mat[:, i] = spl(x_clean)

    if not intercept:
        basis_mat = basis_mat[:, 1:]

    # Penalty matrix
    if fx:
        pen = None
    elif S is None:
        ncol = basis_mat.shape[1] + (0 if intercept else 1)
        D = np.diff(np.eye(ncol), n=diff, axis=0)
        pen = D.T @ D
        pen = (pen + pen.T) / 2
        if not intercept:
            pen = pen[1:, 1:]
    else:
        pen = np.asarray(S, dtype=float)
        if pen.shape[0] != basis_mat.shape[1]:
            raise ValueError("dimensions of 'S' not compatible")

    if nan_mask.any():
        basis_full = np.full((len(x), basis_mat.shape[1]), np.nan)
        basis_full[~nan_mask] = basis_mat[~nan_mask]
        basis_mat = basis_full

    return OneBasis(
        basis_mat,
        fun="ps",
        df=df,
        knots=knots,
        degree=degree,
        intercept=intercept,
        fx=fx,
        S=pen,
        diff_order=diff,
    )


# ---------------------------------------------------------------------------
# Natural cubic regression splines (cr)
# ---------------------------------------------------------------------------

@_register("cr")
def cr(
    x: np.ndarray,
    df: int = 10,
    knots: Optional[np.ndarray] = None,
    intercept: bool = False,
    fx: bool = False,
    S: Optional[np.ndarray] = None,
    **kwargs,
) -> OneBasis:
    """Natural cubic regression spline basis (with optional penalty).

    This mirrors the ``cr`` basis from the R ``dlnm`` package, which
    internally uses ``mgcv``'s cubic regression spline.

    Parameters
    ----------
    x : array-like
        Predictor values.
    df : int
        Degrees of freedom (default 10).
    knots : array-like or None
        Knot positions (including boundaries).  If *None*, placed at
        quantiles.
    intercept : bool
        Whether to include an intercept column.
    fx : bool
        If *True*, penalty is not computed.
    S : numpy.ndarray or None
        Custom penalty matrix.

    Returns
    -------
    OneBasis
        CR-spline basis matrix with penalty attribute ``S``.
    """
    x_clean = x.copy()
    nan_mask = np.isnan(x_clean)
    if nan_mask.any():
        x_clean[nan_mask] = 0.0

    if knots is None:
        if df < 3:
            raise ValueError("'df' must be >= 3")
        nk = df + int(not intercept)
        knots = np.quantile(
            np.unique(x_clean[~nan_mask]),
            np.linspace(0, 1, nk),
        )
    else:
        knots = np.sort(np.asarray(knots, dtype=float))
        df = len(knots) - int(not intercept)

    # Build a natural cubic spline basis at the knots
    # Use the approach: basis = ns evaluated at data, knots placed at given positions
    n_knots = len(knots)
    boundary = np.array([knots[0], knots[-1]])
    inner_knots = knots[1:-1]

    basis_mat = _ns_basis(x_clean, inner_knots, boundary, intercept=True)

    # Compute penalty matrix from the basis at knot positions
    basis_at_knots = _ns_basis(knots, inner_knots, boundary, intercept=True)

    # The penalty for a CR spline is based on the integrated squared second derivative
    # We approximate it using the knot-based approach
    n_col = basis_at_knots.shape[1]
    if not fx and S is None:
        # Compute a roughness penalty matrix using finite differences
        h = np.diff(knots)
        # Construct the band matrix for integrated squared second derivative
        n_inner = len(inner_knots)
        # Simple difference penalty as approximation
        if n_col > 2:
            D = np.diff(np.eye(n_col), n=2, axis=0)
            pen = D.T @ D
            pen = (pen + pen.T) / 2
        else:
            pen = np.zeros((n_col, n_col))
    elif S is not None:
        pen = np.asarray(S, dtype=float)
    else:
        pen = None

    if not intercept:
        basis_mat = basis_mat[:, 1:]
        if pen is not None and pen.shape[0] > basis_mat.shape[1]:
            pen = pen[1:, 1:]

    if nan_mask.any():
        basis_full = np.full((len(x), basis_mat.shape[1]), np.nan)
        basis_full[~nan_mask] = basis_mat[~nan_mask]
        basis_mat = basis_full

    return OneBasis(
        basis_mat,
        fun="cr",
        df=df,
        knots=knots,
        intercept=intercept,
        fx=fx,
        S=pen,
    )


# ---------------------------------------------------------------------------
# Strata (indicator variables)
# ---------------------------------------------------------------------------

@_register("strata")
def strata(
    x: np.ndarray,
    df: int = 1,
    breaks: Optional[np.ndarray] = None,
    ref: int = 1,
    intercept: bool = False,
    **kwargs,
) -> OneBasis:
    """Indicator (dummy) basis for strata.

    Parameters
    ----------
    x : array-like
        Predictor values.
    df : int
        Degrees of freedom (determines number of strata if *breaks* is None).
    breaks : array-like or None
        Cut points for creating strata.
    ref : int
        Reference stratum to drop (1-indexed).  Use 0 to keep all.
    intercept : bool
        Whether to prepend an intercept column.

    Returns
    -------
    OneBasis
        Strata indicator matrix.
    """
    rng = np.array([np.nanmin(x), np.nanmax(x)])

    if breaks is not None:
        breaks = np.sort(np.unique(np.asarray(breaks, dtype=float)))
    elif df - int(intercept) > 0:
        n_breaks = df - int(intercept)
        probs = np.linspace(0, 1, n_breaks + 2)[1:-1]
        breaks = np.quantile(x[~np.isnan(x)], probs)
    else:
        breaks = np.array([])

    df = len(breaks) + int(intercept)

    # Cut x into categories
    edges = np.concatenate([[rng[0] - 0.0001], breaks, [rng[1] + 0.0001]])
    # Use digitize (bins are right-open like R's cut with right=FALSE)
    cats = np.digitize(x, edges, right=False) - 1
    cats = np.clip(cats, 0, len(edges) - 2)
    n_levels = len(edges) - 1

    # Create indicator matrix
    basis_mat = np.zeros((len(x), n_levels))
    for i in range(n_levels):
        basis_mat[:, i] = (cats == i).astype(float)

    # Apply reference dropping
    if ref < 0 or ref > basis_mat.shape[1]:
        raise ValueError("wrong value in 'ref' argument")
    if not intercept and ref == 0:
        ref = 1

    if len(breaks) > 0:
        if ref != 0:
            basis_mat = np.delete(basis_mat, ref - 1, axis=1)
        if intercept and ref != 0:
            basis_mat = np.column_stack([np.ones(len(x)), basis_mat])
    # If no breaks, just return what we have (single column)

    return OneBasis(
        basis_mat,
        fun="strata",
        df=df,
        breaks=breaks,
        ref=ref,
        intercept=intercept,
    )


# ---------------------------------------------------------------------------
# Threshold (thr)
# ---------------------------------------------------------------------------

@_register("thr")
def thr(
    x: np.ndarray,
    thr_value: Optional[Union[float, np.ndarray]] = None,
    side: Optional[str] = None,
    intercept: bool = False,
    **kwargs,
) -> OneBasis:
    """Threshold (hockey-stick) basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    thr_value : float, array-like, or None
        Threshold value(s).  Defaults to ``median(x)``.
    side : str or None
        ``"h"`` (high = above threshold), ``"l"`` (low = below threshold),
        or ``"d"`` (double = both sides).  If *None*, inferred from the
        number of threshold values.
    intercept : bool
        Whether to prepend an intercept column.

    Returns
    -------
    OneBasis
        Threshold basis matrix.
    """
    if thr_value is None:
        thr_value = np.nanmedian(x)
    thr_value = np.atleast_1d(np.asarray(thr_value, dtype=float))
    thr_value = np.sort(thr_value)

    if side is None:
        side = "d" if len(thr_value) > 1 else "h"

    if side == "d":
        thr_value = np.array([thr_value[0], thr_value[-1]])
    else:
        thr_value = np.array([thr_value[0]])

    if side not in ("h", "l", "d"):
        raise ValueError("'side' must be one of 'h', 'l', 'd'")

    if side == "h":
        basis_mat = np.maximum(x - thr_value[0], 0).reshape(-1, 1)
    elif side == "l":
        basis_mat = (-np.minimum(x - thr_value[0], 0)).reshape(-1, 1)
    else:  # "d"
        col1 = -np.minimum(x - thr_value[0], 0)
        col2 = np.maximum(x - thr_value[1], 0)
        basis_mat = np.column_stack([col1, col2])

    if intercept:
        basis_mat = np.column_stack([np.ones(len(x)), basis_mat])

    return OneBasis(
        basis_mat,
        fun="thr",
        thr_value=thr_value,
        side=side,
        intercept=intercept,
    )


# ---------------------------------------------------------------------------
# Integer (categorical)
# ---------------------------------------------------------------------------

@_register("integer")
def integer(
    x: np.ndarray,
    values: Optional[np.ndarray] = None,
    intercept: bool = False,
    **kwargs,
) -> OneBasis:
    """Integer / categorical indicator basis.

    Creates dummy variables for each unique integer value of *x*.

    Parameters
    ----------
    x : array-like
        Predictor values (typically integers).
    values : array-like or None
        The set of values to create indicators for.  Defaults to
        ``sorted(unique(x))``.
    intercept : bool
        Whether to keep all columns (if *True*) or drop the first
        (if *False*, the default).

    Returns
    -------
    OneBasis
        Indicator basis matrix.
    """
    if values is None:
        values = np.sort(np.unique(x[~np.isnan(x)]))
    else:
        values = np.asarray(values, dtype=float)

    basis_mat = np.zeros((len(x), len(values)))
    for i, v in enumerate(values):
        basis_mat[:, i] = (x == v).astype(float)

    if basis_mat.shape[1] > 1 and not intercept:
        basis_mat = basis_mat[:, 1:]
    elif basis_mat.shape[1] == 1:
        intercept = True

    return OneBasis(
        basis_mat,
        fun="integer",
        values=values,
        intercept=intercept,
    )


# ---------------------------------------------------------------------------
# Linear (lin)
# ---------------------------------------------------------------------------

@_register("lin")
def lin(x: np.ndarray, intercept: bool = False, **kwargs) -> OneBasis:
    """Simple linear basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    intercept : bool
        Whether to prepend an intercept column.

    Returns
    -------
    OneBasis
        Linear basis matrix (1 or 2 columns).
    """
    basis_mat = x.reshape(-1, 1).copy()
    if intercept:
        basis_mat = np.column_stack([np.ones(len(x)), basis_mat])

    return OneBasis(basis_mat, fun="lin", intercept=intercept)


# ---------------------------------------------------------------------------
# Polynomial (poly)
# ---------------------------------------------------------------------------

@_register("poly")
def poly(
    x: np.ndarray,
    degree: int = 1,
    scale: Optional[float] = None,
    intercept: bool = False,
    **kwargs,
) -> OneBasis:
    """Scaled polynomial basis.

    Parameters
    ----------
    x : array-like
        Predictor values.
    degree : int
        Maximum polynomial degree (default 1).
    scale : float or None
        Scaling factor.  Defaults to ``max(abs(x))``.
    intercept : bool
        If *True*, include degree-0 column.

    Returns
    -------
    OneBasis
        Polynomial basis matrix.
    """
    if scale is None:
        scale = float(np.nanmax(np.abs(x)))
    if scale == 0:
        scale = 1.0

    start = 0 if intercept else 1
    powers = np.arange(start, degree + 1)
    basis_mat = np.column_stack([(x / scale) ** p for p in powers])

    return OneBasis(
        basis_mat,
        fun="poly",
        degree=degree,
        scale=scale,
        intercept=intercept,
    )
