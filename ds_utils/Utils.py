import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable, Any, Dict, List, Tuple, Union


def scale_uniform(series: pd.Series, middle: float = 0.5) -> pd.Series:
    return (series.rank(pct=True, method="first") - middle) / series.shape[0]


def auto_corr_penalty(x: pd.Series, lag: int = 1) -> float:
    n = x.shape[0]
    p = np.abs(x.autocorr(lag=lag))
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


# to neutralize any series by any other series
def neutralize_series(series: pd.Series, by: pd.Series, proportion: float = 1.0) -> pd.Series:
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack((exposures, np.array([series.mean()] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized
