"""Knot placement utilities for DLNMs.

These functions help determine knot positions for spline bases,
particularly for the lag dimension where log-spacing is often
appropriate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pydlnm.utils import mklag


def logknots(
    x: np.ndarray,
    nk: Optional[int] = None,
    fun: str = "ns",
    df: int = 1,
    degree: int = 3,
    intercept: bool = True,
) -> np.ndarray:
    """Place knots at equally-spaced log-transformed values.

    Particularly useful for the lag dimension, where effects typically
    diminish non-linearly with increasing lag.

    Parameters
    ----------
    x : array-like
        If length 1 or 2, interpreted as a lag range.  Otherwise the
        range is computed from the data.
    nk : int or None
        Number of knots.  If *None*, inferred from *fun*, *df*, *degree*,
        and *intercept*.
    fun : str
        Basis function name (``"ns"``, ``"bs"``, ``"strata"``), used to
        determine the number of knots from *df*.
    df : int
        Degrees of freedom.
    degree : int
        Spline degree (for ``"bs"``).
    intercept : bool
        Whether an intercept is included.

    Returns
    -------
    numpy.ndarray
        Array of knot positions.

    Examples
    --------
    >>> from pydlnm import logknots
    >>> logknots(30, nk=3)
    array([ 1.47...,  5.47..., 15.30...])
    """
    x = np.asarray(x, dtype=float).ravel()

    if len(x) < 3:
        rng = mklag(x)
    else:
        rng = np.array([np.nanmin(x), np.nanmax(x)])

    if np.diff(rng)[0] == 0:
        raise ValueError("range must be > 0")

    if nk is None:
        fun = fun.lower()
        if fun == "ns":
            nk = df - 1 - int(intercept)
        elif fun == "bs":
            nk = df - degree - int(intercept)
        elif fun == "strata":
            nk = df - int(intercept)
        else:
            raise ValueError(f"fun must be 'ns', 'bs', or 'strata', got '{fun}'")

    if nk < 1:
        raise ValueError("choice of arguments defines no knots")

    d = float(np.diff(rng)[0])
    knots = rng[0] + np.exp(
        ((1 + np.log(d)) / (nk + 1)) * np.arange(1, nk + 1) - 1
    )

    return knots


def equalknots(
    x: np.ndarray,
    nk: Optional[int] = None,
    fun: str = "ns",
    df: int = 1,
    degree: int = 3,
    intercept: bool = False,
) -> np.ndarray:
    """Place knots at equally-spaced values.

    Parameters
    ----------
    x : array-like
        Data values (the range is used).
    nk : int or None
        Number of knots.  If *None*, inferred from *fun*, *df*, *degree*,
        and *intercept*.
    fun : str
        Basis function name.
    df : int
        Degrees of freedom.
    degree : int
        Spline degree (for ``"bs"``).
    intercept : bool
        Whether an intercept is included.

    Returns
    -------
    numpy.ndarray
        Array of knot positions.

    Examples
    --------
    >>> from pydlnm import equalknots
    >>> equalknots(np.arange(0, 31), nk=3)
    array([ 7.5, 15. , 22.5])
    """
    x = np.asarray(x, dtype=float).ravel()
    rng = np.array([np.nanmin(x), np.nanmax(x)])

    if nk is None:
        fun = fun.lower()
        if fun == "ns":
            nk = df - 1 - int(intercept)
        elif fun == "bs":
            nk = df - degree - int(intercept)
        elif fun == "strata":
            nk = df - int(intercept)
        else:
            raise ValueError(f"fun must be 'ns', 'bs', or 'strata', got '{fun}'")

    if nk < 1:
        raise ValueError("choice of arguments defines no knots")

    d = float(np.diff(rng)[0])
    knots = rng[0] + (d / (nk + 1)) * np.arange(1, nk + 1)

    return knots
