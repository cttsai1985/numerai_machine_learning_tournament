import sys
import os
import json
import re
import joblib
import logging
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from . import DataManager
from .DefaultConfigs import RefreshLevel


class SolutionConfigs:
    def __init__(self):
        pass


class Solution:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: DataManager,
            model: BaseEstimator, fit_params: Optional[Dict[str, Any]] = None,
            cv_splitter: Optional[BaseCrossValidator] = None, scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None):
        self.refresh_level: RefreshLevel = refresh_level
        self.refresh_level_criterion: RefreshLevel = RefreshLevel("model")

        self.is_fitted: bool = False
        self.model: Optional[BaseEstimator] = model
        self.fit_params: Optional[Dict[str, Any]] = fit_params if fit_params is not None else dict()

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter
        self.has_cv_splitter: bool = False
        if cv_splitter is not None:
            self.has_cv_splitter = True

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

    @property
    def model_filepath_(self) -> str:
        return os.path.join(self.working_dir, "model.pkl")

    def save_model(self):
        if not self.is_fitted:
            logging.info(f"cannot save model due to not fitted")
            return self

        logging.info(f"save model to {self.model_filepath_}")
        joblib.dump(deepcopy(self.model), self.model_filepath_)
        return self

    def load_model(self):
        if not (os.path.exists(self.model_filepath_) or os.path.isfile(self.model_filepath_)):
            return self

        logging.info(f"load pre-fitted model from {self.model_filepath_}")
        model = joblib.load(self.model_filepath_)
        if not isinstance(model, type(self.model)):
            logging.info(f"model mismatched: from file ({type(model)}) != model ({type(self.model)})")
            return self

        self.is_fitted = True
        self.model = model
        return self

    def _do_cross_val(self, data_type: Optional[str] = None):
        if not self.has_cv_splitter:
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

    def _do_model_fit(self, data_type: Optional[str] = None):
        if not data_type:
            data_type = "training"

        if self.is_fitted and self.refresh_level > self.refresh_level_criterion:
            logging.info(f"skip due to refresh level ({self.refresh_level}) higher than {self.refresh_level_criterion}")
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_
        self.model.fit(X, y, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self.save_model()
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

        self.load_model()
        self._do_model_fit(data_type=train_data_type)
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self.load_model()
        self._do_model_fit(data_type=train_data_type)
        self._do_inference(data_type=infer_data_type)
        return self

    @classmethod
    def from_configs(cls, args: Namespace, configs: SolutionConfigs, output_data_path: str):
        return cls(
            args.refresh_level, configs.data_manager_, configs.unfitted_model_, scoring_func=configs.scoring_func,
            cv_splitter=configs.cv_splitter_, working_dir=output_data_path, )


if "__main__" == __name__:
    import pdb

    obj = Solution()
    pdb.set_trace()
