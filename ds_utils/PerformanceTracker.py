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
        return pd.Series(
            {k: v["func"](**{"y_true": y_true, "y_pred": y_pred, "sample_weight": sample_weight}) for k, v in
             self.metrics.items() if (scoring_type is None or scoring_type == v["type"])}, name="score")


if "__main__" == __name__:
    import pdb

    obj = PerformanceTracker()
    pdb.set_trace()
