import sys
import os
import json
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from . import DataManager


class SolutionConfigs:
    def __init__(self):
        pass


class Solution:
    def __init__(
            self, model: BaseEstimator, data_manager: DataManager, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None):
        self.is_fitted: bool = False
        self.model: BaseEstimator = model
        self.fit_params: Optional[Dict] = dict()

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter
        self.can_eval: bool = False
        if cv_splitter is not None:
            self.can_eval = True

        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.data_manager: DataManager = data_manager

        self.scoring_func: Callable = scoring_func

        self.cross_val_predictions: Optional[pd.DataFrame] = None
        self.cross_val_score_all: Optional[pd.DataFrame] = None
        self.cross_val_score_split: Optional[pd.DataFrame] = None
        self.cross_val_fnc_all: Optional[pd.DataFrame] = None
        self.cross_val_fnc_split: Optional[pd.DataFrame] = None

        self.valid_predictions: Optional[pd.DataFrame] = None
        self.valid_score_all: Optional[pd.DataFrame] = None
        self.valid_score_split: Optional[pd.DataFrame] = None
        self.valid_fnc_all: Optional[pd.DataFrame] = None
        self.valid_fnc_split: Optional[pd.DataFrame] = None

    def _do_cross_val(self, data_type: Optional[str] = None):
        if not self.can_eval:
            return self

        if not data_type:
            data_type = "training"

        cross_val_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = cross_val_data.data_
        yhat = cross_val_predict(
            self.model, X, y=y, cv=self.cv_splitter, n_jobs=None, verbose=0, fit_params=self.fit_params,
            pre_dispatch='2*n_jobs', method='predict')
        yhat = pd.Series(yhat, index=y.index, name="yhat")

        ret = cross_val_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)

        self.cross_val_predictions, self.cross_val_score_all, self.cross_val_score_split = ret
        self.cross_val_predictions.to_parquet(os.path.join(self.working_dir, "cross_val_predictions.parquet"))
        self.cross_val_score_all.to_parquet(os.path.join(self.working_dir, "cross_val_score_all.parquet"))
        self.cross_val_score_split.to_parquet(os.path.join(self.working_dir, "cross_val_score_split.parquet"))

        ret = cross_val_data.evaluate(yhat=yhat, scoring_func=self.scoring_func, feature_neutral=True)
        _, self.cross_val_fnc_all, self.cross_val_fnc_split = ret
        self.cross_val_fnc_all.to_parquet(os.path.join(self.working_dir, "cross_val_fnc_all.parquet"))
        self.cross_val_fnc_split.to_parquet(os.path.join(self.working_dir, "cross_val_fnc_split.parquet"))
        return self

    def _do_model_fit(self, data_type: Optional[str] = None, refit: bool = False):
        if not data_type:
            data_type = "training"

        # TODO: save / restore model status
        if self.is_fitted and not refit:
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_
        self.model.fit(X, y, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        return self

    def _do_validation(self, data_type: Optional[str] = None):
        if data_type == "skip":
            return self

        if not data_type:
            data_type = "validation"

        if not self.is_fitted:
            logging.warning(f"model is not fitted yet, skip")
            return self

        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = valid_data.data_

        yhat = pd.Series(self.model.predict(X), index=y.index, name="yhat")
        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)

        self.valid_predictions, self.valid_score_all, self.valid_score_split = ret
        self.valid_predictions.to_parquet(os.path.join(self.working_dir, "validation_predictions.parquet"))
        self.valid_score_all.to_parquet(os.path.join(self.working_dir, "validation_score_all.parquet"))
        self.valid_score_split.to_parquet(os.path.join(self.working_dir, "validation_score_split.parquet"))

        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func, feature_neutral=True)
        _, self.valid_fnc_all, self.valid_fnc_split = ret
        self.valid_fnc_all.to_parquet(os.path.join(self.working_dir, "validation_fnc_all.parquet"))
        self.valid_fnc_split.to_parquet(os.path.join(self.working_dir, "validation_fnc_split.parquet"))
        return self

    def _do_inference(self, data_type: str = "tournament", ):
        logging.info(f"inference on {data_type}")
        if not self.is_fitted:
            logging.warning(f"model is not fitted yet, skip")
            return self

        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = infer_data.data_

        ret = pd.DataFrame({"prediction": self.model.predict(X), "id": y.index}).reindex(columns=["id", "prediction"])
        ret.to_parquet(os.path.join(self.working_dir, f"{data_type}_predictions.parquet"))
        ret.to_csv(os.path.join(self.working_dir, f"{data_type}_predictions.csv"), index=False)
        return self

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_cross_val(data_type=train_data_type)
        self._do_model_fit(data_type=train_data_type)
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._do_model_fit(data_type=train_data_type)
        self._do_inference(data_type=infer_data_type)
        return self


if "__main__" == __name__:
    import pdb

    obj = Solution()
    pdb.set_trace()
