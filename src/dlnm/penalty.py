"""Penalty matrix construction for penalised DLNMs.

:func:`cb_pen` constructs the list of penalty matrices for a cross-basis
or one-basis, to be used with penalised regression methods.
"""

from __future__ import annotations

import numpy as np

from dlnm.utils import findrank


def cb_pen(
    cb,
    sp: float | np.ndarray = -1,
    addSlag: np.ndarray | list | None = None,
) -> dict:
    """Build penalty matrices for a cross-basis or one-basis.

    Parameters
    ----------
    cb : CrossBasis or OneBasis
        The basis object.
    sp : float or array-like
        Smoothing parameter(s).  ``-1`` means to be estimated (default).
    addSlag : numpy.ndarray, list, or None
        Additional penalty matrices for the lag dimension.

    Returns
    -------
    dict
        Dictionary with penalty matrices (``"Svar"``, ``"Slag"``),
        ``"rank"``, and ``"sp"`` entries.

    Raises
    ------
    ValueError
        If *cb* is not a valid basis, or dimensions are inconsistent.
    """
    from dlnm.basis import OneBasis
    from dlnm.crossbasis import CrossBasis

    is_onebasis = isinstance(cb, OneBasis)
    is_crossbasis = isinstance(cb, CrossBasis)

    if not is_onebasis and not is_crossbasis:
        raise TypeError("first argument must be a CrossBasis or OneBasis")

    if is_onebasis:
        # Transform to cross-basis-like structure
        ncol = cb.shape[1]
        df = np.array([ncol, 1])
        argvar_fun = cb.fun
        arglag_fun = "strata"
        argvar_fx = getattr(cb, "fx", True) or argvar_fun not in ("ps", "cr")
        arglag_fx = True
        Svar = getattr(cb, "S", None)
        Slag = None
    else:
        df = cb.df
        argvar_fun = cb.argvar.get("fun", "ns")
        arglag_fun = cb.arglag.get("fun", "strata")
        argvar_fx = argvar_fun not in ("ps", "cr") or cb.argvar.get("fx", False)
        arglag_fx = arglag_fun not in ("ps", "cr") or cb.arglag.get("fx", False)
        Svar = cb.argvar.get("S", None)
        Slag = cb.arglag.get("S", None)

    Slist = {}

    if not argvar_fx and Svar is not None:
        Slist["Svar"] = np.kron(Svar, np.eye(int(df[1])))

    if not arglag_fx and Slag is not None:
        Slist["Slag"] = np.kron(np.eye(int(df[0])), Slag)

    # Rescale penalties
    for key in list(Slist.keys()):
        ev = np.linalg.eigvalsh(Slist[key])
        max_ev: float = float(np.max(ev))
        if max_ev > 0:
            Slist[key] = Slist[key] / max_ev

    # Additional lag penalties
    if is_onebasis and addSlag is not None:
        raise ValueError("penalties on lag not allowed for OneBasis")

    if addSlag is not None:
        add_list = _mkaddSlag(addSlag, df)
        Slist.update(add_list)

    # Rank
    rank = {k: findrank(v) for k, v in Slist.items()}

    # Smoothing parameters
    npen = len(Slist)
    if npen == 0:
        raise ValueError("no penalisation defined")

    sp_arr: np.ndarray = np.atleast_1d(np.asarray(sp, dtype=float))  # type: ignore[assignment]
    if sp_arr.size == 1:
        sp_arr = np.repeat(sp_arr, npen)
    if sp_arr.size != npen:
        raise ValueError("'sp' must be consistent with number of penalty terms")

    result = dict(Slist)
    result["rank"] = rank
    result["sp"] = sp_arr

    return result


def _mkaddSlag(addSlag, df: np.ndarray) -> dict:
    """Build additional penalty matrices for the lag dimension.

    Parameters
    ----------
    addSlag : array-like or list of arrays
        Penalty matrix/matrices for the lag dimension.
    df : numpy.ndarray
        ``[df_var, df_lag]``.

    Returns
    -------
    dict
        Named penalty matrices.
    """
    if not isinstance(addSlag, list):
        addSlag = [addSlag]

    Slist = {}
    for i, S in enumerate(addSlag):
        S = np.asarray(S, dtype=float)
        if S.ndim == 1:
            S = np.diag(S)
        if S.shape[0] != int(df[1]) or S.shape[1] != int(df[1]):
            raise ValueError("addSlag dimensions not consistent with lag basis")

        # Rescale
        ev = np.linalg.eigvalsh(S)
        max_ev2: float = float(np.max(ev))
        if max_ev2 > 0:
            S = S / max_ev2

        # Expand: I(df_var) kron S
        expanded = np.kron(np.eye(int(df[0])), S)
        Slist[f"Slag{i + 2}"] = expanded

    return Slist
