"""Internal utility functions for PyDLNM."""

from __future__ import annotations

import numpy as np


def mklag(lag: int | list | np.ndarray) -> np.ndarray:
    """Normalise a lag specification to a two-element integer array ``[lag_min, lag_max]``.

    Parameters
    ----------
    lag : int or array-like of length 1 or 2
        If a single integer, interpreted as ``[0, lag]`` when non-negative,
        or ``[lag, 0]`` when negative.  If length-2, used directly after
        validation.

    Returns
    -------
    numpy.ndarray
        Two-element integer array ``[lag_min, lag_max]``.

    Raises
    ------
    ValueError
        If *lag* is not numeric or has length > 2, or if ``lag[0] > lag[1]``.
    """
    lag_arr = np.atleast_1d(np.asarray(lag, dtype=float)).ravel()
    if lag_arr.size > 2:
        raise ValueError("'lag' must be an integer scalar or length-2 vector")
    if lag_arr.size == 1:
        lag_arr = np.array([lag_arr[0], 0]) if lag_arr[0] < 0 else np.array([0, lag_arr[0]])
    if lag_arr[1] - lag_arr[0] < 0:
        raise ValueError("lag[0] must be <= lag[1]")
    return np.round(lag_arr[:2]).astype(int)


def seqlag(lag: np.ndarray, by: int = 1) -> np.ndarray:
    """Generate a sequence of lag values from *lag[0]* to *lag[1]*.

    Parameters
    ----------
    lag : array-like
        Two-element lag range ``[lag_min, lag_max]``.
    by : int, optional
        Step size (default 1).

    Returns
    -------
    numpy.ndarray
        Sequence of lag integers.
    """
    lag = np.asarray(lag)
    return np.arange(lag[0], lag[1] + by, by)


def mkat(
    at: np.ndarray | list | None = None,
    from_val: float | None = None,
    to_val: float | None = None,
    by: float | None = None,
    range_val: np.ndarray | None = None,
    lag: np.ndarray | None = None,
    bylag: int = 1,
) -> np.ndarray:
    """Build the vector of predictor values for which to compute predictions.

    Parameters
    ----------
    at : array-like or None
        Explicit predictor values.  If a 2-D array, each row is a set of
        lagged exposure values.
    from_val, to_val : float or None
        Lower / upper bounds (default to range limits).
    by : float or None
        Step size.  If *None*, ~50 pretty values are generated.
    range_val : array-like
        ``[min, max]`` of the original predictor.
    lag : array-like
        Lag range.
    bylag : int
        Lag step.

    Returns
    -------
    numpy.ndarray
        1-D array of predictor values, or 2-D matrix if *at* is a matrix.
    """
    if at is None:
        fv: float = from_val if from_val is not None else float(range_val[0])  # type: ignore[index]
        tv: float = to_val if to_val is not None else float(range_val[1])  # type: ignore[index]
        if by is None:
            nobs = 50
            at = np.linspace(fv, tv, nobs + 1)
            # Make "pretty" values
            nice = _pretty(np.array([fv, tv]), n=nobs)
            nice = nice[(nice >= fv) & (nice <= tv)]
            if len(nice) > 0:
                at = nice
        else:
            nice = _pretty(np.array([fv, tv]), n=max(1, int((tv - fv) / by)))
            nice = nice[(nice >= fv) & (nice <= tv)]
            if len(nice) > 0:
                at = np.arange(nice[0], tv + by * 0.5, by)
                at = at[at <= tv]  # type: ignore[operator]
            else:
                at = np.arange(fv, tv + by * 0.5, by)
    elif isinstance(at, np.ndarray) and at.ndim == 2:
        lag_len = int(lag[1] - lag[0]) + 1  # type: ignore[index]
        if at.shape[1] != lag_len:
            raise ValueError(f"matrix 'at' must have ncol=diff(lag)+1={lag_len}")
        if bylag != 1:
            raise ValueError("'bylag!=1 not allowed with 'at' in matrix form")
    else:
        at = np.sort(np.unique(np.asarray(at, dtype=float)))
    return at  # type: ignore[return-value]


def _pretty(x: np.ndarray, n: int = 5) -> np.ndarray:
    """Attempt to emulate R's ``pretty()`` for generating nice axis ticks."""
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return np.array([lo])
    import matplotlib.ticker as ticker

    loc = ticker.MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10])
    ticks = loc.tick_values(lo, hi)
    return np.asarray(ticks)


def mkcen(
    cen: float | bool | None = None,
    type_: str = "cb",
    basis=None,
    range_val: np.ndarray | None = None,
) -> float | None:
    """Determine the centering value for predictions.

    Parameters
    ----------
    cen : float, bool, or None
        User-supplied centering value.  ``True`` = auto, ``False`` = none.
    type_ : str
        One of ``"cb"``, ``"one"``.
    basis : OneBasis or CrossBasis
        The basis object carrying metadata.
    range_val : array-like
        ``[min, max]`` of the original predictor.

    Returns
    -------
    float or None
        The centering value, or *None* if centering should not be applied.
    """
    nocen = cen is None

    # Try to extract from basis if not supplied
    if nocen:
        if type_ == "cb":
            cen = getattr(basis, "argvar", {}).get("cen", None)
        elif type_ == "one":
            cen = getattr(basis, "cen", None)

    # Determine the basis function name
    if type_ == "cb":
        fun = getattr(basis, "argvar", {}).get("fun", None)
    elif type_ == "one":
        fun = getattr(basis, "fun", None)
    else:
        fun = None

    # For certain functions, centering is not meaningful
    if fun in ("thr", "strata", "integer", "lin"):
        if isinstance(cen, bool):
            cen = None
    else:
        # If None or True, set to approx mid-range
        if cen is None or (isinstance(cen, bool) and cen):
            pretty_vals = _pretty(np.asarray(range_val))
            cen = float(np.median(pretty_vals))
        # If explicitly False, no centering
        if isinstance(cen, bool) and not cen:
            cen = None

    # If intercept present, no centering needed
    if type_ == "cb":
        intercept = getattr(basis, "argvar", {}).get("intercept", False)
    elif type_ == "one":
        intercept = getattr(basis, "intercept", False)
    else:
        intercept = False
    if intercept is True:
        cen = None

    if nocen and cen is not None:
        import warnings

        warnings.warn(
            f"centering value unspecified. Automatically set to {cen}",
            stacklevel=3,
        )

    return cen


def lag_matrix(x: np.ndarray, lag: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
    """Create a matrix of lagged values from a time-series vector.

    Equivalent to ``tsModel::Lag`` in R.

    Parameters
    ----------
    x : numpy.ndarray
        1-D time-series.
    lag : numpy.ndarray
        Sequence of lag values (e.g. ``[0, 1, 2, ..., L]``).
    group : numpy.ndarray or None
        Optional group indicator for panel data.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(x), len(lag))`` with lagged values.
    """
    n = len(x)
    nlags = len(lag)
    result = np.full((n, nlags), np.nan)

    if group is None:
        for j, l in enumerate(lag):
            l = int(l)
            if l >= 0:
                result[l:, j] = x[: n - l]
            else:
                result[: n + l, j] = x[-l:]
    else:
        groups = np.unique(group)
        for g in groups:
            idx = np.where(group == g)[0]
            xg = x[idx]
            ng = len(xg)
            for j, l in enumerate(lag):
                l = int(l)
                if l >= 0:
                    result[idx[l:], j] = xg[: ng - l]
                else:
                    result[idx[: ng + l], j] = xg[-l:]
    return result


def tensor_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise tensor (Kronecker) product of two matrices.

    Equivalent to ``mgcv::tensor.prod.model.matrix``.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix of shape ``(n, p)``.
    B : numpy.ndarray
        Matrix of shape ``(n, q)``.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(n, p*q)`` where row *i* is ``kron(A[i,:], B[i,:])``.
    """
    n = A.shape[0]
    p = A.shape[1]
    q = B.shape[1]
    result = np.empty((n, p * q))
    for j in range(p):
        result[:, j * q : (j + 1) * q] = A[:, j : j + 1] * B
    return result


def exphist(
    exp: np.ndarray,
    times: np.ndarray | None = None,
    lag: int | np.ndarray | None = None,
    fill: float = 0.0,
) -> np.ndarray:
    """Build exposure history matrices from an exposure profile.

    For each time point in *times*, constructs the vector of lagged
    exposures from *exp* going back through the lag period.

    Parameters
    ----------
    exp : array-like
        Exposure profile (1-D vector).
    times : array-like or None
        Time points at which to evaluate exposure histories.
        Defaults to ``range(1, len(exp)+1)``.
    lag : int, array-like, or None
        Lag specification (see :func:`mklag`).  Defaults to
        ``[0, len(exp)-1]``.
    fill : float, optional
        Value to use for out-of-range lags (default 0).

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(times), lag_period)`` where each row is
        the exposure history for the corresponding time point.
    """
    exp = np.asarray(exp, dtype=float).ravel()

    lag = np.array([0, len(exp) - 1]) if lag is None else mklag(lag)

    if times is None:
        times = np.arange(1, len(exp) + 1)
    else:
        times = np.round(np.asarray(times, dtype=float)).astype(int)

    # Extend exp on both sides
    left = max(0, int(lag[1]) + 1 - int(times.min()))  # type: ignore[union-attr]
    right = max(0, int(times.max()) - len(exp) - int(lag[0]))  # type: ignore[union-attr]
    exp_ext = np.concatenate([np.full(left, fill), exp, np.full(right, fill)])

    lag_seq = seqlag(lag)
    n_lags = len(lag_seq)
    hist = np.empty((len(times), n_lags))  # type: ignore[arg-type]

    for i, t in enumerate(times):  # type: ignore[union-attr,arg-type]
        start = int(t - lag[0] + left) - 1  # 0-indexed
        for j, l in enumerate(lag_seq):
            idx = start - int(l)
            if 0 <= idx < len(exp_ext):
                hist[i, j] = exp_ext[idx]
            else:
                hist[i, j] = fill

    return hist


def findrank(X: np.ndarray) -> int:
    """Find the numerical rank of a symmetric matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Symmetric matrix.

    Returns
    -------
    int
        Estimated rank.
    """
    ev = np.linalg.eigvalsh(X)
    return int(np.sum(ev > np.max(ev) * np.finfo(float).eps * 10))
