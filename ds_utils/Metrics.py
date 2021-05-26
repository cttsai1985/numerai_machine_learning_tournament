import logging
import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr


def corr(y_true: Union[np.array, pd.Series], y_pred: pd.Series, func: Callable, **kwargs) -> float:
    if func == spearmanr:
        y_pred = y_pred.rank(pct=True, method="first")

    return func(y_true, y_pred)[0]  # correlation


def median_absolute_error_fixed(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    return metrics.median_absolute_error(y_true, y_pred)


# Payout is just the score cliped at +/-25%
def payout(scores: pd.Series) -> float:
    return scores.clip(lower=-0.25, upper=0.25)


spearman_corr = partial(corr, func=spearmanr)
pearson_corr = partial(corr, func=pearsonr)

available_metrics: Dict[str, Dict[str, Union[str, Callable]]] = {
    "RMSE": {"type": "error", "func": partial(metrics.mean_squared_error, squared=False)},
    "MAE": {"type": "error", "func": metrics.mean_absolute_error},
    "MedAE": {"type": "error", "func": median_absolute_error_fixed},
    # "MAPE": {"type": "error", "func": metrics.mean_absolute_percentage_error),
    "Spearman": {"type": "corr", "func": spearman_corr},
    "Pearson": {"type": "corr", "func": pearson_corr},
}
