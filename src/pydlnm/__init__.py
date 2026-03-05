"""
PyDLNM - Distributed Lag Non-Linear Models for Python.

A Python port of the R ``dlnm`` package by Antonio Gasparrini.

This package provides functions to specify and interpret distributed lag
linear and non-linear models (DLNMs). DLNMs represent a modelling framework
for describing simultaneously the non-linear and delayed effects of
exposures on health outcomes.

References
----------
Gasparrini A. Distributed lag linear and non-linear models in R: the
package dlnm. *Journal of Statistical Software*. 2011; 43(8):1-20.
https://doi.org/10.18637/jss.v043.i08

Original R package: https://github.com/gasparrini/dlnm
"""

from pydlnm.basis import (
    cr,
    integer,
    lin,
    ns,
    bs,
    onebasis,
    poly,
    ps,
    strata,
    thr,
)
from pydlnm.crossbasis import crossbasis
from pydlnm.crosspred import crosspred
from pydlnm.crossreduce import crossreduce
from pydlnm.datasets import load_chicagoNMMAPS, load_drug, load_nested
from pydlnm.knots import equalknots, logknots
from pydlnm.penalty import cb_pen
from pydlnm.plotting import plot_crosspred, plot_crossreduce
from pydlnm.utils import exphist, seqlag

__version__ = "0.1.0"

__all__ = [
    # Core
    "onebasis",
    "crossbasis",
    "crosspred",
    "crossreduce",
    # Basis functions
    "ns",
    "bs",
    "ps",
    "cr",
    "strata",
    "thr",
    "integer",
    "lin",
    "poly",
    # Knot placement
    "logknots",
    "equalknots",
    # Penalty
    "cb_pen",
    # Plotting
    "plot_crosspred",
    "plot_crossreduce",
    # Utilities
    "exphist",
    "seqlag",
    # Datasets
    "load_chicagoNMMAPS",
    "load_drug",
    "load_nested",
]
