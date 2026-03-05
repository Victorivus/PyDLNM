"""Plotting functions for DLNMs.

Provides visualisation of cross-predictions and cross-reductions
including 3-D surface plots, contour plots, slice plots, and overall
cumulative effect plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from pydlnm.utils import seqlag


def plot_crosspred(
    pred,
    ptype: str | None = None,
    var: float | list | None = None,
    lag: int | list | None = None,
    ci: str = "area",
    ci_level: float | None = None,
    cumul: bool = False,
    exp: bool | None = None,
    title: str | None = None,
    figsize: tuple = (8, 6),
    ax=None,
    **kwargs,
):
    """Plot predictions from a fitted DLNM.

    Parameters
    ----------
    pred : CrossPred
        A prediction object from :func:`crosspred`.
    ptype : str or None
        Plot type: ``"3d"``, ``"contour"``, ``"slices"``, or ``"overall"``.
        Automatically determined if *None*.
    var : float, list of float, or None
        Exposure value(s) at which to plot lag-response curves
        (``ptype="slices"``).
    lag : int, list of int, or None
        Lag value(s) at which to plot exposure-response curves
        (``ptype="slices"``).
    ci : str
        Confidence interval style: ``"area"``, ``"bars"``, ``"lines"``,
        or ``"n"`` (none).
    ci_level : float or None
        Confidence level (defaults to that in *pred*).
    cumul : bool
        If *True*, plot cumulative effects (requires ``cumul=True`` in
        :func:`crosspred`).
    exp : bool or None
        If *True*, exponentiate effects.  Auto-detected from link.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib axes or None
        Axes to plot on (creates new figure if *None*).
    **kwargs
        Additional keyword arguments passed to the matplotlib plot command.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if ci_level is None:
        ci_level = pred.ci_level
    z = norm.ppf(1 - (1 - ci_level) / 2)

    # Determine plot type
    if ptype is None:
        if var is not None or lag is not None:
            ptype = "slices"
        elif np.diff(pred.lag)[0] == 0:
            ptype = "overall"
        else:
            ptype = "3d"

    # Work with copies
    matfit = pred.cumfit.copy() if cumul else pred.matfit.copy()
    matse = pred.cumse.copy() if cumul else pred.matse.copy()
    allfit = pred.allfit.copy()
    allse = pred.allse.copy()

    if cumul and pred.cumfit is None:
        raise ValueError("Cumulative outcomes require cumul=True in crosspred()")

    mathigh = matfit + z * matse
    matlow = matfit - z * matse
    allhigh = allfit + z * allse
    alllow = allfit - z * allse
    noeff = 0

    # Exponentiate?
    do_exp = False
    if exp is True or exp is None and pred.model_link in ("log", "logit"):
        do_exp = True

    if do_exp:
        matfit = np.exp(matfit)
        mathigh = np.exp(mathigh)
        matlow = np.exp(matlow)
        allfit = np.exp(allfit)
        allhigh = np.exp(allhigh)
        alllow = np.exp(alllow)
        noeff = 1

    predvar = pred.predvar
    lag_seq = seqlag(pred.lag, pred.bylag)

    # --- SLICES ---
    if ptype == "slices":
        if var is None and lag is None:
            raise ValueError("at least 'var' or 'lag' must be provided for ptype='slices'")
        n_plots = (len(np.atleast_1d(lag)) if lag is not None else 0) + (
            len(np.atleast_1d(var)) if var is not None else 0
        )

        fig, _axes_raw = plt.subplots(
            1, n_plots, figsize=(figsize[0] * n_plots / 2, figsize[1]), squeeze=False
        )
        axes: Any = _axes_raw.ravel()
        plot_idx = 0

        # Lag-specific (exposure-response at fixed lag)
        if lag is not None:
            for l_val in np.atleast_1d(lag):
                l_idx = np.where(lag_seq == l_val)[0]
                if len(l_idx) == 0:
                    raise ValueError(f"lag={l_val} not in prediction lags")
                l_idx = l_idx[0]
                ax_cur = axes[plot_idx]
                _plot_ci(
                    ax_cur,
                    predvar,
                    matfit[:, l_idx],
                    mathigh[:, l_idx],
                    matlow[:, l_idx],
                    ci,
                    noeff,
                    **kwargs,
                )
                ax_cur.set_xlabel("Var")
                ax_cur.set_ylabel("Effect")
                ax_cur.set_title(f"Lag = {l_val}" if title is None else title)
                plot_idx += 1

        # Variable-specific (lag-response at fixed exposure)
        if var is not None:
            for v_val in np.atleast_1d(var):
                v_idx = np.where(np.isclose(predvar, v_val))[0]
                if len(v_idx) == 0:
                    v_idx = np.array([np.argmin(np.abs(predvar - v_val))])
                v_idx = v_idx[0]
                ax_cur = axes[plot_idx]
                _plot_ci(
                    ax_cur,
                    lag_seq,
                    matfit[v_idx, :],
                    mathigh[v_idx, :],
                    matlow[v_idx, :],
                    ci,
                    noeff,
                    **kwargs,
                )
                ax_cur.set_xlabel("Lag")
                ax_cur.set_ylabel("Effect")
                ax_cur.set_title(f"Var = {v_val}" if title is None else title)
                plot_idx += 1

        fig.tight_layout()
        return fig

    # --- OVERALL ---
    if ptype == "overall":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        _plot_ci(ax, predvar, allfit, allhigh, alllow, ci, noeff, **kwargs)
        ax.set_xlabel("Var")
        ax.set_ylabel("Effect")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    # --- CONTOUR ---
    if ptype == "contour":
        fig, ax_new = plt.subplots(figsize=figsize)
        levels = np.linspace(matfit.min(), matfit.max(), 20)
        cf = ax_new.contourf(predvar, lag_seq, matfit.T, levels=levels, cmap="RdBu_r")
        fig.colorbar(cf, ax=ax_new)
        ax_new.set_xlabel("Var")
        ax_new.set_ylabel("Lag")
        if title:
            ax_new.set_title(title)
        fig.tight_layout()
        return fig

    # --- 3D ---
    if ptype == "3d":
        fig = plt.figure(figsize=figsize)
        ax3 = fig.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(predvar, lag_seq)
        ax3.plot_surface(X, Y, matfit.T, cmap="coolwarm", alpha=0.85, edgecolor="none")
        ax3.set_xlabel("Var")
        ax3.set_ylabel("Lag")
        ax3.set_zlabel("Effect")
        ax3.view_init(elev=30, azim=210)
        if title:
            ax3.set_title(title)
        fig.tight_layout()
        return fig

    raise ValueError(f"Unknown ptype '{ptype}'")


def _plot_ci(ax, x, fit, high, low, ci, noeff, **kwargs):
    """Helper to plot a line with confidence intervals."""
    if ci == "area":
        ax.fill_between(x, low, high, alpha=0.2, color="grey")
    elif ci == "bars":
        rng = (x.max() - x.min()) / 300
        ax.vlines(x, low, high, colors="grey", linewidth=0.5)
        for xi, lo, hi in zip(x, low, high):
            ax.hlines(hi, xi - rng, xi + rng, colors="grey", linewidth=0.5)
            ax.hlines(lo, xi - rng, xi + rng, colors="grey", linewidth=0.5)
    elif ci == "lines":
        ax.plot(x, high, linestyle="--", color="grey", linewidth=0.8)
        ax.plot(x, low, linestyle="--", color="grey", linewidth=0.8)

    ax.plot(x, fit, **kwargs)
    ax.axhline(noeff, color="black", linewidth=0.5, linestyle=":")


def plot_crossreduce(
    red,
    ci: str = "area",
    ci_level: float | None = None,
    exp: bool | None = None,
    title: str | None = None,
    figsize: tuple = (8, 6),
    ax=None,
    **kwargs,
):
    """Plot a cross-reduction.

    Parameters
    ----------
    red : CrossReduce
        A reduction object from :func:`crossreduce`.
    ci : str
        Confidence interval style.
    ci_level : float or None
        Confidence level.
    exp : bool or None
        Whether to exponentiate.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib axes or None
        Axes to plot on.
    **kwargs
        Additional keyword arguments passed to the matplotlib plot command.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    if ci_level is None:
        ci_level = red.ci_level
    z = norm.ppf(1 - (1 - ci_level) / 2)

    fit = red.fit.copy()
    high = fit + z * red.se
    low = fit - z * red.se
    noeff = 0

    do_exp = False
    if exp is True or exp is None and red.model_link in ("log", "logit"):
        do_exp = True

    if do_exp:
        fit = np.exp(fit)
        high = np.exp(high)
        low = np.exp(low)
        noeff = 1

    if red.type == "var":
        xvar = seqlag(red.lag, red.bylag)
        xlabel = "Lag"
    else:
        xvar = red.predvar
        xlabel = "Var"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _plot_ci(ax, xvar, fit, high, low, ci, noeff, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Effect")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
