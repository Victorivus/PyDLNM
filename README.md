# PyDLNM — Distributed Lag Non-Linear Models for Python

[CI](https://github.com/Victorivus/PyDLNM/actions/workflows/ci.yml)
[PyPI version](https://pypi.org/project/pydlnm/)
[Python](https://pypi.org/project/pydlnm/)
[License: GPL v2+](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)

A Python implementation of **Distributed Lag Non-Linear Models (DLNMs)**, based on the widely-used R package `[dlnm](https://github.com/gasparrini/dlnm)` by Antonio Gasparrini, Ben Armstrong, and Fabian Scheipl.

DLNMs are a modelling framework for describing simultaneously the **non-linear** and **delayed** effects of exposures on health outcomes. They are extensively used in environmental epidemiology, pharmacology, and other fields studying lagged exposure-response relationships.

## Attribution

This package is a **Python port** of the original R package:

> **Gasparrini A.** Distributed lag linear and non-linear models in R: the package dlnm.
> *Journal of Statistical Software*. 2011; 43(8):1–20.
> [doi:10.18637/jss.v043.i08](https://doi.org/10.18637/jss.v043.i08)

Original R source code: [https://github.com/gasparrini/dlnm](https://github.com/gasparrini/dlnm)

All core algorithms and methodology are from the original R implementation. If you use this package in academic work, please cite the original publication. Acknowledging this Python adaptation in your citations is also appreciated.

## Installation

```bash
pip install pydlnm
```

For development:

```bash
git clone https://github.com/Victorivus/PyDLNM.git
cd PyDLNM
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import statsmodels.api as sm
from pydlnm import crossbasis, crosspred, logknots, load_chicagoNMMAPS

# Load example data (Chicago NMMAPS)
df = load_chicagoNMMAPS()

# Define the cross-basis for temperature with 21-day lag
lknots = logknots(21, nk=3)
cb_temp = crossbasis(
    df["temp"].values,
    lag=21,
    argvar={"fun": "ns", "df": 4},
    arglag={"fun": "ns", "knots": lknots},
)

# Fit a model (e.g., quasi-Poisson GLM)
# Build design matrix with cross-basis + covariates
time = df["time"].values
X = np.column_stack([np.ones(len(df)), cb_temp, time])
y = df["death"].values

# Use coefficients and vcov directly
# (or extract from a fitted statsmodels model)
```

## Features

### Basis Functions


| Function  | Description                                     |
| --------- | ----------------------------------------------- |
| `ns`      | Natural cubic splines                           |
| `bs`      | B-splines                                       |
| `ps`      | P-splines (penalised B-splines)                 |
| `cr`      | Natural cubic regression splines (with penalty) |
| `strata`  | Indicator (dummy) variables                     |
| `thr`     | Threshold / hockey-stick                        |
| `integer` | Categorical coding for integer values           |
| `lin`     | Simple linear term                              |
| `poly`    | Scaled polynomial basis                         |


### Core Functions

- `**onebasis(x, fun, ...)**` — Create a single-dimension basis matrix
- `**crossbasis(x, lag, argvar, arglag)**` — Create a cross-basis matrix (tensor product of variable and lag bases)
- `**crosspred(basis, model, ...)**` — Compute predictions (lag-specific, overall, cumulative)
- `**crossreduce(basis, model, type, ...)**` — Reduce to uni-dimensional summaries

### Utilities

- `**logknots(x, nk)**` — Log-spaced knot placement (ideal for lag dimension)
- `**equalknots(x, nk)**` — Equally-spaced knot placement
- `**exphist(exp, times, lag)**` — Build exposure history matrices
- `**cb_pen(cb)**` — Penalty matrices for penalised regression

### Plotting

- `**plot_crosspred(pred, ptype)**` — 3D surfaces, contour plots, slices, overall effects
- `**plot_crossreduce(red)**` — Plot reduced uni-dimensional associations

### Datasets

- `**load_chicagoNMMAPS()**` — Daily mortality/weather/pollution data, Chicago 1987–2000
- `**load_drug()**` — Simulated RCT with time-varying drug doses
- `**load_nested()**` — Simulated nested case-control study

## Usage Examples

### Exposure-Response at a Specific Lag

```python
from pydlnm import crossbasis, crosspred, plot_crosspred

# After fitting a model...
pred = crosspred(cb, coef=coef, vcov=vcov, at=np.arange(-20, 35), cen=21)
fig = plot_crosspred(pred, ptype="slices", lag=0)  # Exposure-response at lag 0
```

### Lag-Response at a Specific Exposure

```python
fig = plot_crosspred(pred, ptype="slices", var=30)  # Lag-response at 30°C
```

### 3D Surface Plot

```python
fig = plot_crosspred(pred, ptype="3d")
```

### Overall Cumulative Effect

```python
fig = plot_crosspred(pred, ptype="overall")
```

### Reduced to One Dimension

```python
from pydlnm import crossreduce, plot_crossreduce

# Overall cumulative exposure-response
red_overall = crossreduce(cb, coef=coef, vcov=vcov, type="overall", cen=21)
fig = plot_crossreduce(red_overall)

# Lag-response at a specific exposure value
red_var = crossreduce(cb, coef=coef, vcov=vcov, type="var", value=30, cen=21)
fig = plot_crossreduce(red_var)
```

## R Compatibility Notes

PyDLNM is designed to produce results numerically identical to R's `dlnm` package (within floating-point tolerance). Several non-obvious implementation choices were required to match R's behaviour exactly. These are documented here for maintainers and contributors.

### 1. Natural spline basis — linear extrapolation outside boundary knots

**Location:** `[src/pydlnm/basis.py](src/pydlnm/basis.py)`, `_ns_basis()`

R's `splines::ns()` extrapolates **linearly** beyond the boundary knots (zero second derivative, by the natural spline constraint). scipy's `BSpline(extrapolate=True)` continues the cubic polynomial instead, producing completely different values outside the boundary. This matters whenever prediction points fall outside the training data range (a common case for `crosspred`).

**Fix:** After computing the B-spline basis values, points outside `[boundary_knots[0], boundary_knots[1]]` are replaced with a linear extrapolation anchored at the boundary value and first derivative:

```python
if left_mask.any():
    val_l  = spl(boundary_knots[0])
    slope_l = spl(boundary_knots[0], 1)
    basis[left_mask, i] = val_l + slope_l * (x[left_mask] - boundary_knots[0])
```

### 2. Natural spline basis — `intercept=False` constraint row

**Location:** `[src/pydlnm/basis.py](src/pydlnm/basis.py)`, `_ns_basis()`

R excludes the "intercept" B-spline (B-spline 0, the leftmost) by adding the unit vector `e0 = [1, 0, 0, …]` as a third constraint row before the QR null-space projection, not an all-ones row.

**Fix:**

```python
if not intercept:
    e0 = np.zeros((1, n_basis))
    e0[0, 0] = 1.0
    const = np.vstack([const, e0])
```

### 3. `crosspred` — prediction grid ordering (varvec / lagvec)

**Location:** `[src/pydlnm/crosspred.py](src/pydlnm/crosspred.py)`, `_mkXpred()`

R builds the flat prediction vector as `rep(predvar, length(predlag))` (the full predvar sequence repeated once per lag) with `lagvec = rep(predlag, each=length(predvar))`. This means **predvar varies fastest** (inner loop), predlag slowest (outer loop). The opposite ordering breaks both the allfit accumulation loop and the matfit reshape.

**Fix:**

```python
varvec = np.tile(predvar, len(predlag))       # predvar repeats for each lag
lagvec = np.repeat(predlag, len(predvar))     # each lag repeated len(predvar) times
```

### 4. `crosspred` — matfit / matse reshape order

**Location:** `[src/pydlnm/crosspred.py](src/pydlnm/crosspred.py)`, `crosspred()`

With the corrected predvar-fastest ordering (point 3), the flat `Xpred @ coef` vector is laid out as `[predvar@lag0, predvar@lag1, …]`. Reshaping it directly as `(n_pred, n_lag)` gives the wrong column assignments. The correct reshape matches R's column-major (Fortran-order) matrix filling:

```python
matfit = (Xpred @ coef).reshape(len(predlag), len(predvar)).T
matse  = np.sqrt(...).reshape(len(predlag), len(predvar)).T
```

### 5. Generating R golden-reference fixtures

The test suite compares Python output against fixtures produced by R. To regenerate them, run the script from the **PyDLNM repository root**:

```bash
Rscript tests/fixtures/generate_fixtures.R
```

On Windows, `Rscript` may not be in `PATH`. Use the full path:

```bat
Rscript tests\fixtures\generate_fixtures.R
```

---

## Differences from the R Package


| Feature             | R `dlnm`                                    | Python `pydlnm`                                             |
| ------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| Model fitting       | Integrated with `glm`, `gam`, `coxph`, etc. | Works with `statsmodels` or manual `coef`/`vcov`            |
| Penalised smoothing | Via `mgcv::gam`                             | Penalty matrices provided via `cb_pen()`                    |
| Plotting            | Base R graphics                             | `matplotlib`                                                |
| S3 classes          | `crossbasis`, `crosspred`, etc.             | Python dataclasses and `numpy.ndarray` subclasses           |
| Function naming     | `crossbasis`, `crosspred`                   | Same names, snake_case helpers (`cb_pen`, `plot_crosspred`) |


## Dependencies

- NumPy >= 1.22
- SciPy >= 1.9
- pandas >= 1.5
- matplotlib >= 3.5
- patsy >= 0.5
- statsmodels >= 0.13

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=pydlnm --cov-report=html

# Lint
ruff check src/ tests/

# Type check
mypy src/pydlnm/
```

## References

1. Gasparrini A. *Distributed lag linear and non-linear models in R: the package dlnm*. Journal of Statistical Software. 2011; 43(8):1–20. [doi:10.18637/jss.v043.i08](https://doi.org/10.18637/jss.v043.i08)
2. Gasparrini A, Armstrong B, Kenward MG. *Distributed lag non-linear models*. Statistics in Medicine. 2010; 29(21):2224–2234. [doi:10.1002/sim.3940](https://doi.org/10.1002/sim.3940)
3. Gasparrini A. *Modeling exposure-lag-response associations with distributed lag non-linear models*. Statistics in Medicine. 2014; 33(5):881–899. [doi:10.1002/sim.5963](https://doi.org/10.1002/sim.5963)

## License

GPL-2.0-or-later — same as the original R package.