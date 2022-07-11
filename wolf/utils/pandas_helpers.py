from pathlib import Path
from typing import List, Callable, Dict, Tuple

import numpy as np
import pandas as pd


def add_bins(df: pd.DataFrame, bins: List[float or int], column: str, pr: int = 1) -> pd.Series:
    """Adds bins for prices for further analysis."""
    labels = [
        f'({bins[i] :.{pr}f}' + f'-{bins[i + 1] :.{pr}f}]' if bins[i] != -1 else '[0' + f'-{bins[i + 1] :.{pr}f}]'
        for i in range(len(bins) - 1)
    ]
    return pd.cut(df[column], bins, labels=labels, ordered=True)


def percentile(n: int) -> Callable:
    """Wrapped function for calculating percentile."""
    assert 0 < n < 100, f'n must be between 0 and 100, got {n}'

    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = f'percentile_{n}'
    return percentile_


def aggregate(df: pd.DataFrame, groupby_columns: List[str], agg_column: str) -> pd.DataFrame:
    """Makes `groupby -> aggregate` for given groupby_columns and agg_column column."""
    aggregations = [
        np.min,
        percentile(5),
        percentile(25),
        percentile(50),
        np.mean,
        percentile(75),
        percentile(95),
        np.max,
        np.std,
        'count'
    ]

    agg_df = df.groupby(groupby_columns).agg({agg_column: aggregations})
    agg_df.columns = [f'{x}_{y}' for (x, y) in agg_df.columns.tolist()]
    return agg_df.reset_index()


def get_iqr_range(x: pd.Series or np.ndarray) -> Tuple[float, float]:
    """Returns the Inter-Quartile Range for given values."""
    q1, q3 = np.percentile(sorted(x), [25, 75])
    iqr = q3 - q1
    return q1 - (1.5 * iqr), q3 + (1.5 * iqr)


def style_table(df: pd.DataFrame, axis: int, precision: int = 0):
    properties = {
        'font-size': '10pt',
        'background-color': 'white',
        'border-color': 'black',
        'border-style': 'solid',
        'border-width': '1px',
        'border-collapse': 'collapse',
        'width': '80px'
    }
    if precision == 0:
        str_format = "{:.0f}"
    elif precision == 1:
        str_format = "{:.1f}"
    elif precision == 2:
        str_format = "{:.2f}"
    elif precision == 3:
        str_format = "{:.3f}"
    elif precision == 4:
        str_format = "{:.4f}"
    else:
        raise ValueError

    return (df
            .style
            .set_properties(**properties)
            .background_gradient(cmap='OrRd', axis=axis)
            .format(str_format, subset=df.select_dtypes(include='number').columns)
            )
