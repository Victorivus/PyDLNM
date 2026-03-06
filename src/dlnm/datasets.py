"""Bundled datasets for examples and testing.

These datasets are ported from the R ``dlnm`` package and include:

- **chicagoNMMAPS**: Daily mortality, weather, and pollution data for
  Chicago (1987--2000) from the National Morbidity, Mortality and Air
  Pollution Study.
- **drug**: Simulated data from a randomised controlled trial with
  time-varying drug doses.
- **nested**: Simulated nested case-control study with time-varying
  occupational exposure and a cancer outcome.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"


def load_chicagoNMMAPS() -> pd.DataFrame:
    """Load the Chicago NMMAPS dataset.

    Daily mortality (all causes, CVD, respiratory), weather
    (temperature, dew point, relative humidity) and pollution
    (PM10, ozone) data for Chicago, 1987--2000.

    Returns
    -------
    pandas.DataFrame
        5 114 rows and 14 columns.

    Examples
    --------
    >>> from dlnm import load_chicagoNMMAPS
    >>> df = load_chicagoNMMAPS()
    >>> df.shape
    (5114, 14)
    >>> df.columns.tolist()[:5]
    ['date', 'time', 'year', 'month', 'doy']
    """
    return pd.read_csv(_DATA_DIR / "chicagoNMMAPS.csv", parse_dates=["date"])


def load_drug() -> pd.DataFrame:
    """Load the simulated drug trial dataset.

    Simulated RCT with 200 subjects receiving time-varying drug doses
    over four weeks, with outcome measured at day 28.

    Returns
    -------
    pandas.DataFrame
        200 rows and 7 columns.

    Examples
    --------
    >>> from dlnm import load_drug
    >>> df = load_drug()
    >>> df.shape
    (200, 7)
    """
    return pd.read_csv(_DATA_DIR / "drug.csv")


def load_nested() -> pd.DataFrame:
    """Load the simulated nested case-control dataset.

    Simulated nested case-control study with 300 risk sets, each
    containing a case and a control matched by age year.

    Returns
    -------
    pandas.DataFrame
        600 rows and 14 columns.

    Examples
    --------
    >>> from dlnm import load_nested
    >>> df = load_nested()
    >>> df.shape
    (600, 14)
    """
    return pd.read_csv(_DATA_DIR / "nested.csv")
