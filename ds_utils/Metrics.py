import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr

from ds_utils import Utils


def _base_corr(y_true: Union[np.array, pd.Series], y_pred: pd.Series, func: Callable, **kwargs) -> float:
    if func == pearsonr:
        y_pred = Utils.scale_uniform(y_pred)

    return func(y_true, y_pred)[0]  # correlation


def spearman_corr(y: np.ndarray, y_preds: np.ndarray, **kwargs) -> float:
    return _base_corr(y, pd.Series(y_preds), func=spearmanr, **kwargs)


def pearson_corr(y: np.ndarray, y_preds: np.ndarray, **kwargs) -> float:
    return _base_corr(y, pd.Series(y_preds), func=pearsonr, **kwargs)


def median_absolute_error_fixed(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    return metrics.median_absolute_error(y_true, y_pred)


metric_spearman_corr = partial(_base_corr, func=spearmanr)
metric_pearson_corr = partial(_base_corr, func=pearsonr)

available_metrics: Dict[str, Dict[str, Union[str, Callable]]] = {
    "RMSE": {"type": "error", "func": partial(metrics.mean_squared_error, squared=False)},
    "MAE": {"type": "error", "func": metrics.mean_absolute_error},
    "MedAE": {"type": "error", "func": median_absolute_error_fixed},
    # "MAPE": {"type": "error", "func": metrics.mean_absolute_percentage_error),
    "Spearman": {"type": "corr", "func": metric_spearman_corr},
    "Pearson": {"type": "corr", "func": metric_pearson_corr},
}
