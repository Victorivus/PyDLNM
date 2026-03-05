"""Predictions from fitted DLNMs.

:func:`crosspred` computes lag-specific effects, overall cumulative effects,
and optionally cumulative effects at each lag, with confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.stats import norm

from pydlnm.basis import OneBasis, onebasis
from pydlnm.utils import mkat, mkcen, mklag, seqlag, tensor_product


@dataclass
class CrossPred:
    """Container for cross-prediction results.

    Attributes
    ----------
    predvar : numpy.ndarray
        Predictor values at which predictions are computed.
    cen : float or None
        Centering value.
    lag : numpy.ndarray
        Lag range ``[min, max]``.
    bylag : int
        Lag step size.
    coefficients : numpy.ndarray
        Model coefficients for the cross-basis terms.
    vcov : numpy.ndarray
        Variance-covariance matrix of coefficients.
    matfit : numpy.ndarray
        Matrix of lag-specific effects (rows=predvar, cols=lags).
    matse : numpy.ndarray
        Matrix of lag-specific standard errors.
    allfit : numpy.ndarray
        Overall (summed) effects for each predictor value.
    allse : numpy.ndarray
        Standard errors of overall effects.
    cumfit : numpy.ndarray or None
        Cumulative effects (if requested).
    cumse : numpy.ndarray or None
        Standard errors of cumulative effects.
    ci_level : float
        Confidence interval level.
    model_class : str or None
        Class of the fitted model.
    model_link : str or None
        Link function of the model.
    matRRfit, matRRlow, matRRhigh : numpy.ndarray or None
        Exponentiated lag-specific effects and CI bounds (for log/logit links).
    allRRfit, allRRlow, allRRhigh : numpy.ndarray or None
        Exponentiated overall effects and CI bounds.
    cumRRfit, cumRRlow, cumRRhigh : numpy.ndarray or None
        Exponentiated cumulative effects and CI bounds.
    matlow, mathigh : numpy.ndarray or None
        Linear-scale CI bounds for lag-specific effects.
    alllow, allhigh : numpy.ndarray or None
        Linear-scale CI bounds for overall effects.
    cumlow, cumhigh : numpy.ndarray or None
        Linear-scale CI bounds for cumulative effects.
    """

    predvar: np.ndarray
    cen: Optional[float] = None
    lag: np.ndarray = field(default_factory=lambda: np.array([0, 0]))
    bylag: int = 1
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    vcov: np.ndarray = field(default_factory=lambda: np.array([]))
    matfit: np.ndarray = field(default_factory=lambda: np.array([]))
    matse: np.ndarray = field(default_factory=lambda: np.array([]))
    allfit: np.ndarray = field(default_factory=lambda: np.array([]))
    allse: np.ndarray = field(default_factory=lambda: np.array([]))
    cumfit: Optional[np.ndarray] = None
    cumse: Optional[np.ndarray] = None
    ci_level: float = 0.95
    model_class: Optional[str] = None
    model_link: Optional[str] = None

    # Exponentiated (log/logit link)
    matRRfit: Optional[np.ndarray] = None
    matRRlow: Optional[np.ndarray] = None
    matRRhigh: Optional[np.ndarray] = None
    allRRfit: Optional[np.ndarray] = None
    allRRlow: Optional[np.ndarray] = None
    allRRhigh: Optional[np.ndarray] = None
    cumRRfit: Optional[np.ndarray] = None
    cumRRlow: Optional[np.ndarray] = None
    cumRRhigh: Optional[np.ndarray] = None

    # Linear-scale
    matlow: Optional[np.ndarray] = None
    mathigh: Optional[np.ndarray] = None
    alllow: Optional[np.ndarray] = None
    allhigh: Optional[np.ndarray] = None
    cumlow: Optional[np.ndarray] = None
    cumhigh: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Return a textual summary of these predictions."""
        lines = [
            "PREDICTIONS:",
            f"values: {len(self.predvar)}",
        ]
        if self.cen is not None:
            lines.append(f"centered at: {self.cen}")
        lines += [
            f"range: {self.predvar.min()}, {self.predvar.max()}",
            f"lag: {self.lag}",
            f"exponentiated: {'yes' if self.allRRfit is not None else 'no'}",
            f"cumulative: {'yes' if self.cumfit is not None else 'no'}",
            "",
            "MODEL:",
            f"parameters: {len(self.coefficients)}",
            f"class: {self.model_class}",
            f"link: {self.model_link}",
        ]
        return "\n".join(lines)


def _mkXpred(type_: str, basis, at, predvar, predlag, cen):
    """Build the design matrix for prediction.

    Parameters
    ----------
    type_ : str
        ``"cb"`` for cross-basis, ``"one"`` for one-basis.
    basis : CrossBasis or OneBasis
        The basis used for fitting.
    at : array-like
        Values at which to predict.
    predvar : array-like
        Predictor values.
    predlag : array-like
        Lag values.
    cen : float or None
        Centering value.

    Returns
    -------
    numpy.ndarray
        Design matrix for prediction.
    """
    if isinstance(at, np.ndarray) and at.ndim == 2:
        varvec = at.ravel()
    else:
        # R ordering: predvar varies fastest (inner), predlag varies slowest (outer)
        # rep(predvar, length(predlag)) in R
        varvec = np.tile(np.asarray(predvar, dtype=float), len(predlag))
    # rep(predlag, each=length(predvar)) in R
    lagvec = np.repeat(np.asarray(predlag, dtype=float), len(predvar))

    if type_ == "cb":
        argvar = dict(basis.argvar)
        argvar.pop("cen", None)
        arglag = dict(basis.arglag)

        basisvar = onebasis(varvec, **argvar)
        basislag = onebasis(lagvec, **arglag)

        if cen is not None:
            basiscen = onebasis(np.array([cen]), **argvar)
            basisvar = basisvar - basiscen

        Xpred = tensor_product(np.asarray(basisvar), np.asarray(basislag))

    elif type_ == "one":
        fun_name = basis.fun
        ob_kwargs = {}
        for attr_name in ("df", "knots", "degree", "intercept", "Boundary_knots",
                          "thr_value", "side", "values", "scale", "breaks", "ref",
                          "S", "fx", "diff_order"):
            val = getattr(basis, attr_name, None)
            if val is not None:
                ob_kwargs[attr_name] = val
        basisvar = onebasis(varvec, fun=fun_name, **ob_kwargs)

        if cen is not None:
            basiscen = onebasis(np.array([cen]), fun=fun_name, **ob_kwargs)
            basisvar = basisvar - basiscen

        Xpred = np.asarray(basisvar)

    return Xpred


def crosspred(
    basis,
    model=None,
    coef: Optional[np.ndarray] = None,
    vcov: Optional[np.ndarray] = None,
    model_link: Optional[str] = None,
    at: Optional[Union[np.ndarray, list]] = None,
    from_val: Optional[float] = None,
    to_val: Optional[float] = None,
    by: Optional[float] = None,
    lag: Optional[Union[int, np.ndarray]] = None,
    bylag: int = 1,
    cen: Optional[Union[float, bool]] = None,
    ci_level: float = 0.95,
    cumul: bool = False,
) -> CrossPred:
    """Compute predictions from a fitted DLNM.

    Extracts (or accepts) model coefficients and their variance-covariance
    matrix, then computes lag-specific, overall, and optionally cumulative
    effects across the exposure-lag space.

    Parameters
    ----------
    basis : CrossBasis or OneBasis
        The cross-basis (or one-basis) used in fitting the model.
    model : fitted model object or None
        A fitted ``statsmodels`` regression model.  If provided, *coef*
        and *vcov* are extracted from it.
    coef : numpy.ndarray or None
        Coefficient vector for the cross-basis terms.  Required if
        *model* is *None*.
    vcov : numpy.ndarray or None
        Variance-covariance matrix.  Required if *model* is *None*.
    model_link : str or None
        Link function name (``"log"``, ``"logit"``, ``"identity"``).
        Auto-detected from *model* when possible.
    at : array-like or None
        Specific predictor values at which to evaluate effects.
    from_val, to_val : float or None
        Range bounds for predictor grid.
    by : float or None
        Step size for predictor grid.
    lag : int, array-like, or None
        Lag sub-range for predictions (defaults to full lag range).
    bylag : int
        Lag step size.
    cen : float, bool, or None
        Centering value.  *True* = auto-detect, *False* = none.
    ci_level : float
        Confidence level (default 0.95).
    cumul : bool
        Whether to compute cumulative effects (default *False*).

    Returns
    -------
    CrossPred
        Object containing all prediction results.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> from pydlnm import crossbasis, crosspred
    >>> # ... fit a model with a crossbasis ...
    >>> # pred = crosspred(cb, model, at=np.arange(-10, 35))
    """
    from pydlnm.crossbasis import CrossBasis

    # Determine type
    if isinstance(basis, CrossBasis):
        type_ = "cb"
    elif isinstance(basis, OneBasis):
        type_ = "one"
    else:
        raise TypeError("'basis' must be a CrossBasis or OneBasis object")

    # Original lag
    if type_ == "cb":
        origlag = basis.lag
    else:
        origlag = np.array([0, 0])

    if lag is None:
        lag = origlag.copy()
    else:
        lag = mklag(lag)

    if not np.array_equal(lag, origlag) and cumul:
        raise ValueError("cumulative prediction not allowed for lag sub-period")

    if not 0 < ci_level < 1:
        raise ValueError("'ci_level' must be between 0 and 1")

    # --- Extract coefficients ---
    if model is not None:
        coef_all, vcov_all, model_link = _extract_from_model(model, model_link)
        # Find cross-basis parameter indices
        n_params = basis.shape[1]
        param_names = _get_param_names(model)
        # Try to match by pattern
        indices = _find_basis_indices(param_names, n_params, basis)
        coef = coef_all[indices]
        vcov = vcov_all[np.ix_(indices, indices)]
        model_class = type(model).__name__
    else:
        if coef is None or vcov is None:
            raise ValueError("At least 'model' or 'coef'+'vcov' must be provided")
        coef = np.asarray(coef, dtype=float).ravel()
        vcov = np.asarray(vcov, dtype=float)
        model_class = None

    # Validate
    npar = basis.shape[1]
    if len(coef) != npar or vcov.shape[0] != npar:
        raise ValueError(
            f"coef/vcov dimensions ({len(coef)}, {vcov.shape}) not consistent "
            f"with basis matrix ({npar} columns)"
        )

    # --- Range and at ---
    range_val = basis.range_ if type_ == "cb" else getattr(basis, "range_", None)
    at_vals = mkat(at, from_val, to_val, by, range_val, lag, bylag)

    predvar = at_vals if at_vals.ndim == 1 else np.arange(at_vals.shape[0]).astype(float)
    predlag = seqlag(lag, bylag)

    # --- Centering ---
    cen_val = mkcen(cen, type_, basis, range_val)

    # --- Lag-specific predictions ---
    Xpred = _mkXpred(type_, basis, at_vals, predvar, predlag, cen_val)

    # R fills matrices column-major (Fortran order): predvar varies fastest.
    # With varvec = tile(predvar, n_lag), lagvec = repeat(predlag, n_pred),
    # the flat vector is [predvar@lag0, predvar@lag1, ...], so reshape as
    # (n_lag, n_pred) then transpose to get (n_pred, n_lag).
    matfit = (Xpred @ coef).reshape(len(predlag), len(predvar)).T
    # Standard errors: sqrt(diag(Xpred @ vcov @ Xpred.T))
    matse = np.sqrt(
        np.maximum(0, np.sum((Xpred @ vcov) * Xpred, axis=1))
    ).reshape(len(predlag), len(predvar)).T

    # --- Overall + cumulative predictions ---
    predlag_int = seqlag(lag)  # integer lags only
    Xpred_all = _mkXpred(type_, basis, at_vals, predvar, predlag_int, cen_val)

    Xpredall = np.zeros((len(predvar), Xpred_all.shape[1]))
    if cumul:
        cumfit = np.zeros((len(predvar), len(predlag_int)))
        cumse = np.zeros((len(predvar), len(predlag_int)))

    for i in range(len(predlag_int)):
        start = len(predvar) * i
        end = len(predvar) * (i + 1)
        Xpredall += Xpred_all[start:end, :]
        if cumul:
            cumfit[:, i] = Xpredall @ coef
            cumse[:, i] = np.sqrt(np.maximum(0, np.sum((Xpredall @ vcov) * Xpredall, axis=1)))

    allfit = (Xpredall @ coef).ravel()
    allse = np.sqrt(np.maximum(0, np.sum((Xpredall @ vcov) * Xpredall, axis=1))).ravel()

    # --- Build result ---
    z = norm.ppf(1 - (1 - ci_level) / 2)

    result = CrossPred(
        predvar=predvar,
        cen=cen_val,
        lag=lag,
        bylag=bylag,
        coefficients=coef,
        vcov=vcov,
        matfit=matfit,
        matse=matse,
        allfit=allfit,
        allse=allse,
        ci_level=ci_level,
        model_class=model_class,
        model_link=model_link,
    )

    if cumul:
        result.cumfit = cumfit
        result.cumse = cumse

    # CI bounds
    if model_link in ("log", "logit"):
        result.matRRfit = np.exp(matfit)
        result.matRRlow = np.exp(matfit - z * matse)
        result.matRRhigh = np.exp(matfit + z * matse)
        result.allRRfit = np.exp(allfit)
        result.allRRlow = np.exp(allfit - z * allse)
        result.allRRhigh = np.exp(allfit + z * allse)
        if cumul:
            result.cumRRfit = np.exp(cumfit)
            result.cumRRlow = np.exp(cumfit - z * cumse)
            result.cumRRhigh = np.exp(cumfit + z * cumse)
    else:
        result.matlow = matfit - z * matse
        result.mathigh = matfit + z * matse
        result.alllow = allfit - z * allse
        result.allhigh = allfit + z * allse
        if cumul:
            result.cumlow = cumfit - z * cumse
            result.cumhigh = cumfit + z * cumse

    return result


# ---------------------------------------------------------------------------
# Helpers for extracting from statsmodels
# ---------------------------------------------------------------------------

def _extract_from_model(model, model_link=None):
    """Extract coefficients, vcov, and link from a statsmodels model."""
    coef = np.asarray(model.params).ravel()
    vcov = np.asarray(model.cov_params())

    if model_link is not None:
        return coef, vcov, model_link

    # Try to detect link
    link = "identity"
    if hasattr(model, "family"):
        family = model.family
        if hasattr(family, "link"):
            link_obj = family.link
            link_name = type(link_obj).__name__.lower()
            if "log" in link_name:
                link = "log"
            elif "logit" in link_name:
                link = "logit"
            else:
                link = "identity"
    elif hasattr(model, "model") and hasattr(model.model, "family"):
        family = model.model.family
        if hasattr(family, "link"):
            link_obj = family.link
            link_name = type(link_obj).__name__.lower()
            if "log" in link_name:
                link = "log"
            elif "logit" in link_name:
                link = "logit"

    return coef, vcov, link


def _get_param_names(model):
    """Get parameter names from a model."""
    if hasattr(model, "params") and hasattr(model.params, "index"):
        return list(model.params.index)
    elif hasattr(model, "model") and hasattr(model.model, "exog_names"):
        return model.model.exog_names
    return [f"x{i}" for i in range(len(model.params))]


def _find_basis_indices(param_names, n_params, basis):
    """Find the indices of cross-basis parameters in the model.

    Searches by pattern matching on the parameter names.  Falls back to
    taking the last *n_params* parameters if no match is found.
    """
    import re

    # Try to find by cross-basis naming pattern (v1.l1, v1.l2, etc.)
    pattern = re.compile(r"v\d+\.l\d+")
    indices = [i for i, name in enumerate(param_names) if pattern.search(name)]

    if len(indices) == n_params:
        return np.array(indices)

    # Try suffix matching for onebasis (b1, b2, ...)
    pattern = re.compile(r"b\d+$")
    indices = [i for i, name in enumerate(param_names) if pattern.search(name)]
    if len(indices) == n_params:
        return np.array(indices)

    # Fallback: assume the cross-basis columns are the last n_params
    total = len(param_names)
    if total >= n_params:
        return np.arange(total - n_params, total)

    raise ValueError(
        f"Cannot find {n_params} cross-basis parameters in model "
        f"with {total} parameters"
    )
