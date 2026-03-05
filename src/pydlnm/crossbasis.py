"""Cross-basis matrix construction for DLNMs.

The cross-basis is the core data structure in a DLNM.  It represents the
bi-dimensional exposure-lag-response relationship through a tensor product
of two marginal bases: one for the exposure dimension and one for the lag
dimension.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from pydlnm.basis import onebasis
from pydlnm.utils import lag_matrix, mklag, seqlag


class CrossBasis(np.ndarray):
    """A cross-basis matrix with metadata.

    Subclasses :class:`numpy.ndarray` and carries extra information about
    the variable and lag basis specifications, lag period, and data range.
    """

    _metadata = ["df", "range_", "lag", "argvar", "arglag", "group_count"]

    df: np.ndarray | None
    range_: np.ndarray | None
    lag: np.ndarray | None
    argvar: dict | None
    arglag: dict | None
    group_count: int | None

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
        return (
            f"CrossBasis(shape={self.shape}, "
            f"df_var={self.df[0] if self.df else '?'}, "
            f"df_lag={self.df[1] if self.df else '?'}, "
            f"lag={self.lag})"
        )

    def summary(self) -> str:
        """Return a textual summary of this cross-basis."""
        lines = [
            "CROSSBASIS FUNCTIONS",
            f"observations: {self.shape[0]}",
            f"range: {self.range_[0]} to {self.range_[1]}",  # type: ignore[index]
            f"lag period: {self.lag}",
            f"total df: {self.shape[1]}",
            "",
            "BASIS FOR VAR:",
        ]
        if self.argvar:
            for k, v in self.argvar.items():
                if k != "cen":
                    lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("BASIS FOR LAG:")
        if self.arglag:
            for k, v in self.arglag.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def crossbasis(
    x: np.ndarray | pd.Series,
    lag: int | np.ndarray | None = None,
    argvar: dict | None = None,
    arglag: dict | None = None,
    group: np.ndarray | None = None,
) -> CrossBasis:
    """Construct a cross-basis matrix for a DLNM.

    The cross-basis is the core data structure of a DLNM.  It encodes
    the bi-dimensional relationship between an exposure variable and a
    lag dimension through a tensor product of two marginal basis
    transformations.

    Parameters
    ----------
    x : array-like
        Exposure variable.  A 1-D vector is treated as a time series;
        a 2-D matrix is treated as pre-computed lagged occurrences
        (rows = observations, columns = lags).
    lag : int, array-like, or None
        Lag specification.  A single integer *L* is expanded to ``[0, L]``.
        If *None* and *x* is a matrix, inferred as ``[0, ncol(x)-1]``.
    argvar : dict or None
        Arguments for the variable basis (passed to :func:`onebasis`).
        Must include ``"fun"`` (default ``"ns"``).
    arglag : dict or None
        Arguments for the lag basis (passed to :func:`onebasis`).
        Defaults to ``{"fun": "strata", "df": 1, "intercept": True}``
        when the lag period is zero or when empty.
    group : array-like or None
        Group indicator for panel (non-contiguous) time-series data.

    Returns
    -------
    CrossBasis
        Cross-basis matrix of shape ``(n, df_var * df_lag)`` with
        metadata attributes.

    Examples
    --------
    >>> import numpy as np
    >>> from pydlnm import crossbasis
    >>> np.random.seed(42)
    >>> temp = np.random.randn(500) * 5 + 20
    >>> cb = crossbasis(temp, lag=21,
    ...                 argvar={"fun": "ns", "df": 4},
    ...                 arglag={"fun": "ns", "knots": np.array([1, 5, 14])})
    >>> cb.shape
    (500, 16)
    """
    if argvar is None:
        argvar = {}
    else:
        argvar = dict(argvar)  # copy
    if arglag is None:
        arglag = {}
    else:
        arglag = dict(arglag)  # copy

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    dim = x.shape

    # Normalise lag
    lag = np.array([0, dim[1] - 1]) if lag is None else mklag(lag)

    lag_period = int(lag[1] - lag[0]) + 1
    if dim[1] not in (1, lag_period):
        raise ValueError(
            f"NCOL(x) must be 1 (time series) or {lag_period} (lagged matrix), got {dim[1]}"
        )

    # --- Variable basis ---
    argvar_call = dict(argvar)
    basisvar = onebasis(x.ravel(), **argvar_call)

    # --- Lag basis ---
    # Defaults when arglag is empty or lag period is 0
    if len(arglag) == 0 or int(lag[1] - lag[0]) == 0:
        arglag = {"fun": "strata", "df": 1, "intercept": True}

    # Default: include intercept for lag basis
    arglag.get("fun", "ns")
    if "intercept" not in arglag:
        arglag["intercept"] = True
    # Force no centering for lag basis
    arglag.pop("cen", None)

    lag_seq = seqlag(lag)
    basislag = onebasis(lag_seq.astype(float), **arglag)

    # --- Tensor product ---
    ncol_var = basisvar.shape[1]
    ncol_lag = basislag.shape[1]

    if group is not None:
        group = np.asarray(group)
        if dim[1] > 1:
            raise ValueError("'group' allowed only for time series data")
        min_group_len = min(len(np.where(group == g)[0]) for g in np.unique(group))
        if min_group_len <= int(lag[1] - lag[0]):
            raise ValueError("each group must have length > diff(lag)")

    cb = np.full((dim[0], ncol_var * ncol_lag), np.nan)
    for v in range(ncol_var):
        if dim[1] == 1:
            # Time-series: create lagged column (first lag rows will be NaN)
            mat = lag_matrix(basisvar[:, v], lag_seq, group=group)
        else:
            # Pre-lagged matrix: reshape basis values
            mat = basisvar[:, v].reshape(dim[0], lag_period)
        for l in range(ncol_lag):
            col_idx = ncol_lag * v + l
            cb[:, col_idx] = np.sum(mat * basislag[:, l], axis=1)

    # --- Column names ---
    [f"v{v + 1}.l{l + 1}" for v in range(ncol_var) for l in range(ncol_lag)]

    # --- Reconstruct argvar/arglag from basis attributes ---
    argvar_out: dict[str, object] = {"fun": basisvar.fun}
    for attr_name in (
        "df",
        "knots",
        "degree",
        "intercept",
        "Boundary_knots",
        "thr_value",
        "side",
        "values",
        "scale",
        "breaks",
        "ref",
        "S",
        "fx",
        "diff_order",
    ):
        val = getattr(basisvar, attr_name, None)
        if val is not None:
            argvar_out[attr_name] = val
    argvar_out["cen"] = basisvar.cen

    arglag_out: dict[str, object] = {"fun": basislag.fun}
    for attr_name in (
        "df",
        "knots",
        "degree",
        "intercept",
        "Boundary_knots",
        "thr_value",
        "side",
        "values",
        "scale",
        "breaks",
        "ref",
        "S",
        "fx",
        "diff_order",
    ):
        val = getattr(basislag, attr_name, None)
        if val is not None:
            arglag_out[attr_name] = val

    result = CrossBasis(
        cb,
        df=np.array([ncol_var, ncol_lag]),
        range_=np.array([np.nanmin(x), np.nanmax(x)]),
        lag=lag,
        argvar=argvar_out,
        arglag=arglag_out,
    )
    if group is not None:
        result.group_count = len(np.unique(group))

    return result
