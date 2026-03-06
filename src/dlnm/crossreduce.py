"""Reduction of fitted DLNMs to uni-dimensional summaries.

:func:`crossreduce` reduces the bi-dimensional cross-basis fit to a
single dimension — overall cumulative, lag-specific at a given lag,
or variable-specific at a given exposure value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from dlnm.basis import OneBasis, onebasis
from dlnm.crossbasis import CrossBasis
from dlnm.utils import mkat, mkcen, mklag, seqlag


@dataclass
class CrossReduce:
    """Container for cross-reduction results.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Reduced coefficients.
    vcov : numpy.ndarray
        Reduced variance-covariance matrix.
    basis : OneBasis
        The reduced basis.
    type : str
        Reduction type (``"overall"``, ``"var"``, or ``"lag"``).
    value : float or None
        The fixed value used for ``"var"`` or ``"lag"`` reductions.
    predvar : numpy.ndarray or None
        Predictor values (for ``"overall"`` or ``"lag"``).
    cen : float or None
        Centering value.
    lag : numpy.ndarray
        Lag range.
    bylag : int
        Lag step.
    fit : numpy.ndarray
        Fitted values on the linear predictor scale.
    se : numpy.ndarray
        Standard errors.
    ci_level : float
        Confidence interval level.
    model_class : str or None
        Model class name.
    model_link : str or None
        Link function name.
    RRfit, RRlow, RRhigh : numpy.ndarray or None
        Exponentiated effects and CI (for log/logit).
    low, high : numpy.ndarray or None
        Linear-scale CI bounds.
    """

    coefficients: np.ndarray
    vcov: np.ndarray
    basis: OneBasis
    type: str
    value: float | None = None
    predvar: np.ndarray | None = None
    cen: float | None = None
    lag: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    bylag: int = 1
    fit: np.ndarray = field(default_factory=lambda: np.array([]))
    se: np.ndarray = field(default_factory=lambda: np.array([]))
    ci_level: float = 0.95
    model_class: str | None = None
    model_link: str | None = None

    RRfit: np.ndarray | None = None
    RRlow: np.ndarray | None = None
    RRhigh: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def summary(self) -> str:
        """Return a textual summary."""
        lines = [
            "REDUCED FIT",
            f"type: {self.type}",
            f"dimension: {'predictor' if self.type != 'var' else 'lag'}",
        ]
        if self.type != "overall":
            lines.append(f"value: {self.value}")
        lines.append(f"reduced df: {len(self.coefficients)}")
        return "\n".join(lines)


def crossreduce(
    basis: CrossBasis,
    model=None,
    type: str = "overall",
    value: float | None = None,
    coef: np.ndarray | None = None,
    vcov: np.ndarray | None = None,
    model_link: str | None = None,
    at: np.ndarray | list | None = None,
    from_val: float | None = None,
    to_val: float | None = None,
    by: float | None = None,
    lag: int | np.ndarray | None = None,
    bylag: int = 1,
    cen: float | bool | None = None,
    ci_level: float = 0.95,
) -> CrossReduce:
    """Reduce a cross-basis fit to a uni-dimensional summary.

    Parameters
    ----------
    basis : CrossBasis
        The cross-basis used in fitting.
    model : fitted model or None
        A fitted ``statsmodels`` model.
    type : str
        Reduction type: ``"overall"`` (sum over lags), ``"lag"``
        (effect at a fixed lag), or ``"var"`` (lag-response at a fixed
        exposure value).
    value : float or None
        The fixed value for ``"var"`` or ``"lag"`` reductions.
    coef, vcov : numpy.ndarray or None
        Model coefficients and variance-covariance.
    model_link : str or None
        Link function.
    at : array-like or None
        Predictor values.
    from_val, to_val, by : float or None
        Grid specification.
    lag : int, array-like, or None
        Lag sub-range.
    bylag : int
        Lag step.
    cen : float, bool, or None
        Centering value.
    ci_level : float
        Confidence level.

    Returns
    -------
    CrossReduce
        Object containing the reduced predictions.
    """
    if not isinstance(basis, CrossBasis):
        raise TypeError("basis must be a CrossBasis object")

    type = type.lower()
    if type not in ("overall", "var", "lag"):
        raise ValueError("'type' must be 'overall', 'var', or 'lag'")

    if type != "overall":
        if value is None:
            raise ValueError("'value' must be provided for type 'var' or 'lag'")
        if type == "lag" and (value < basis.lag[0] or value > basis.lag[1]):  # type: ignore[index]
            raise ValueError("'value' of lag-specific effects must be within the lag range")

    lag = basis.lag.copy() if lag is None else mklag(lag)  # type: ignore[union-attr]

    if not 0 < ci_level < 1:
        raise ValueError("'ci_level' must be between 0 and 1")

    # --- Extract coefficients ---
    if model is not None:
        from dlnm.crosspred import _extract_from_model, _find_basis_indices, _get_param_names

        coef_all, vcov_all, model_link = _extract_from_model(model, model_link)
        param_names = _get_param_names(model)
        indices = _find_basis_indices(param_names, basis.shape[1], basis)
        coef = coef_all[indices]
        vcov = vcov_all[np.ix_(indices, indices)]
        model_class = type.__class__.__name__ if hasattr(type, "__class__") else None
        model_class = model.__class__.__name__
    else:
        if coef is None or vcov is None:
            raise ValueError("At least 'model' or 'coef'+'vcov' must be provided")
        coef = np.asarray(coef, dtype=float).ravel()
        vcov = np.asarray(vcov, dtype=float)
        model_class = None

    # --- at and centering ---
    if at is not None and isinstance(at, np.ndarray) and at.ndim == 2:
        raise ValueError("argument 'at' must be a vector")
    range_val = basis.range_
    at_vals = mkat(at, from_val, to_val, by, range_val, lag, bylag)

    cen_val = mkcen(cen, "cb", basis, range_val)

    # --- Reduction ---
    argvar = dict(basis.argvar)  # type: ignore[arg-type]
    argvar.pop("cen", None)
    arglag = dict(basis.arglag)  # type: ignore[arg-type]

    ncol_var = int(basis.df[0])  # type: ignore[index]
    ncol_lag = int(basis.df[1])  # type: ignore[index]

    if type == "overall":
        lagbasis = onebasis(seqlag(lag).astype(float), **arglag)
        # M = I(ncol_var) kron (1' @ lagbasis)
        ones = np.ones((1, int(lag[1] - lag[0]) + 1))
        M_lag = ones @ np.asarray(lagbasis)  # shape: (1, ncol_lag)
        M = np.kron(np.eye(ncol_var), M_lag)

        newbasis = onebasis(at_vals.astype(float), **argvar)
        if cen_val is not None:
            basiscen = onebasis(np.array([cen_val]), **argvar)
            newbasis = OneBasis(np.asarray(newbasis) - np.asarray(basiscen))
            newbasis.fun = argvar.get("fun", "ns")

    elif type == "lag":
        lagbasis = onebasis(np.array([value], dtype=float), **arglag)
        M = np.kron(np.eye(ncol_var), np.asarray(lagbasis))

        newbasis = onebasis(at_vals.astype(float), **argvar)
        if cen_val is not None:
            basiscen = onebasis(np.array([cen_val]), **argvar)
            newbasis = OneBasis(np.asarray(newbasis) - np.asarray(basiscen))
            newbasis.fun = argvar.get("fun", "ns")

    elif type == "var":
        varbasis = onebasis(np.array([value], dtype=float), **argvar)
        if cen_val is not None:
            basiscen = onebasis(np.array([cen_val]), **argvar)
            varbasis = OneBasis(np.asarray(varbasis) - np.asarray(basiscen))
        M = np.kron(np.asarray(varbasis), np.eye(ncol_lag))

        newbasis = onebasis(seqlag(lag, bylag).astype(float), **arglag)

    # New reduced coefficients and vcov
    newcoef = (M @ coef).ravel()
    newvcov = M @ vcov @ M.T

    # Predictions
    nb = np.asarray(newbasis)
    fit = (nb @ newcoef).ravel()
    se = np.sqrt(np.maximum(0, np.sum((nb @ newvcov) * nb, axis=1))).ravel()

    # --- Build result ---
    z = norm.ppf(1 - (1 - ci_level) / 2)

    result = CrossReduce(
        coefficients=newcoef,
        vcov=newvcov,
        basis=newbasis,
        type=type,
        value=value if type != "overall" else None,
        lag=lag,
        bylag=bylag,
        fit=fit,
        se=se,
        ci_level=ci_level,
        model_class=model_class,
        model_link=model_link,
    )

    if type != "var":
        result.predvar = at_vals
    if cen_val is not None:
        result.cen = cen_val

    if model_link in ("log", "logit"):
        result.RRfit = np.exp(fit)
        result.RRlow = np.exp(fit - z * se)
        result.RRhigh = np.exp(fit + z * se)
    else:
        result.low = fit - z * se
        result.high = fit + z * se

    return result
