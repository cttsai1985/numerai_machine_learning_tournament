import sys
import os
import json
import re
import logging
import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr


def corr(y_true: np.array, y_pred: np.array, func: Callable, **kwargs) -> float:
    return func(y_true, y_pred)[0]  # correlation


def median_absolute_error_fixed(y_true: np.array, y_pred: np.array, **kwargs) -> float:
    return metrics.median_absolute_error(y_true, y_pred)


available_metrics: Dict[str, Dict[str, Union[str, Callable]]] = {
    "RMSE": {"type": "error", "func": partial(metrics.mean_squared_error, squared=False)},
    "MAE": {"type": "error", "func": metrics.mean_absolute_error},
    "MedAE": {"type": "error", "func": median_absolute_error_fixed},
    # "MAPE": {"type": "error", "func": metrics.mean_absolute_percentage_error),
    "Spearman": {"type": "corr", "func": partial(corr, func=spearmanr)},
    "Pearson": {"type": "corr", "func": partial(corr, func=pearsonr)},
}


class PerformanceTracker:
    def __init__(self, eval_metrics: Optional[List[str]] = None):
        eval_metrics = available_metrics.keys() if not eval_metrics else eval_metrics
        self.metrics = {k: available_metrics[k] for k in eval_metrics if k in available_metrics}
        logging.info(f"Using metrics: {self.metrics.keys()}")

    def score(
            self, y_true: pd.Series, y_pred: pd.Series, sample_weight: Optional[pd.Series] = None,
            scoring_type: str = None):
        return pd.Series(
            {k: v["func"](**{"y_true": y_true, "y_pred": y_pred, "sample_weight": sample_weight}) for k, v in
             self.metrics.items() if (scoring_type is None or scoring_type == v["type"])}, name="score")


if "__main__" == __name__:
    import pdb

    obj = PerformanceTracker()
    pdb.set_trace()
