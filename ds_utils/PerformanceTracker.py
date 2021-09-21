import logging
import pandas as pd
from typing import Optional, List

from ds_utils.Metrics import available_metrics


class PerformanceTracker:
    def __init__(self, eval_metrics: Optional[List[str]] = None):
        eval_metrics = available_metrics.keys() if not eval_metrics else eval_metrics
        self.metrics = {k: available_metrics[k] for k in eval_metrics if k in available_metrics}
        logging.info(f"Using metrics: {self.metrics.keys()}")

    def score(
            self, y_true: pd.Series, y_pred: pd.Series, sample_weight: Optional[pd.Series] = None,
            scoring_type: str = None):

        metrics = self.metrics
        if scoring_type is not None:
            metrics = {k: v for k, v in self.metrics.items() if scoring_type == v["type"]}

        mask = y_true.notna()
        if not mask.all():
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]

        data = {"y_true": y_true, "y_pred": y_pred, "sample_weight": sample_weight}
        series = pd.Series({k: v["func"](**data) for k, v in metrics.items()}, name="score")
        series.index.name = "score"
        return series


if "__main__" == __name__:
    import pdb

    obj = PerformanceTracker()
    pdb.set_trace()
