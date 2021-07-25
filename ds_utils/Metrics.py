import logging
import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr


def scale_uniform(series: pd.Series, middle: float = 0.5) -> pd.Series:
    return (series.rank(pct=True, method="first") - middle) / series.shape[0]


def corr(y_true: Union[np.array, pd.Series], y_pred: pd.Series, func: Callable, **kwargs) -> float:
    if func == spearmanr:
        y_pred = scale_uniform(y_pred)

    return func(y_true, y_pred)[0]  # correlation


def spearman_corr(y: np.ndarray, y_preds: np.ndarray, **kwargs) -> float:
    return corr(y, pd.Series(y_preds), func=spearmanr, **kwargs)


def pearson_corr(y: np.ndarray, y_preds: np.ndarray, **kwargs) -> float:
    return corr(y, pd.Series(y_preds), func=pearsonr, **kwargs)


def median_absolute_error_fixed(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    return metrics.median_absolute_error(y_true, y_pred)


# Payout is just the score cliped at +/-25%
def payout(scores: pd.Series, lower: float = -0.25, upper: float = .25) -> float:
    return scores.clip(lower=lower, upper=upper)


metric_spearman_corr = partial(corr, func=spearmanr)
metric_pearson_corr = partial(corr, func=pearsonr)

available_metrics: Dict[str, Dict[str, Union[str, Callable]]] = {
    "RMSE": {"type": "error", "func": partial(metrics.mean_squared_error, squared=False)},
    "MAE": {"type": "error", "func": metrics.mean_absolute_error},
    "MedAE": {"type": "error", "func": median_absolute_error_fixed},
    # "MAPE": {"type": "error", "func": metrics.mean_absolute_percentage_error),
    "Spearman": {"type": "corr", "func": metric_spearman_corr},
    "Pearson": {"type": "corr", "func": metric_pearson_corr},
}
