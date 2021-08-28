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

from .DefaultConfigs import RefreshLevel
from .Utils import scale_uniform


class BaseSolution:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):
        self.refresh_level: RefreshLevel = refresh_level
        self.refresh_level_criterion: RefreshLevel = RefreshLevel("model")

        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.data_manager: "DataManager" = data_manager

        self.scoring_func: Callable = scoring_func

        self.round_digit: int = 3

        self.valid_predictions: Optional[pd.DataFrame] = None
        self.valid_score_all: Optional[pd.DataFrame] = None
        self.valid_score_split: Optional[pd.DataFrame] = None
        self.valid_fnc_all: Optional[pd.DataFrame] = None
        self.valid_fnc_split: Optional[pd.DataFrame] = None

    def _cast_for_classifier_fit(self, target: pd.Series) -> pd.Series:
        cast_type = "label2index"
        if self.data_manager.has_cast_mapping(cast_type):
            return self.data_manager.cast_target(target, cast_type)

        return target

    def _cast_for_classifier_predict(self, target: pd.Series) -> pd.Series:
        cast_type = "index2label"
        if self.data_manager.has_cast_mapping(cast_type):
            return self.data_manager.cast_target(target, cast_type)

        return target

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError

    def _save_prediction(self, eval_type: str = "cross_val"):
        self.cross_val_predictions.to_parquet(
            os.path.join(self.working_dir, f"{eval_type}_predictions.parquet"))
        self.cross_val_score_split.to_parquet(os.path.join(self.working_dir, "cross_val_score_split.parquet"))
        return self

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError

    def _do_validation(self, data_type: Optional[str] = None):
        raise NotImplementedError

    def _do_inference(self, data_type: str = "tournament", ):
        raise NotImplementedError

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
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        raise NotImplementedError


class Solution(BaseSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):

        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)

        self.is_fitted: bool = False
        self.model: Optional[BaseEstimator] = model
        self.fit_params: Optional[Dict[str, Any]] = fit_params if fit_params is not None else dict()

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter
        self.has_cv_splitter: bool = False
        if cv_splitter is not None:
            self.has_cv_splitter = True

        self.cross_val_predictions: Optional[pd.DataFrame] = None
        self.cross_val_score_all: Optional[pd.DataFrame] = None
        self.cross_val_score_split: Optional[pd.DataFrame] = None
        self.cross_val_fnc_all: Optional[pd.DataFrame] = None
        self.cross_val_fnc_split: Optional[pd.DataFrame] = None

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

        y = self._cast_for_classifier_fit(y)
        yhat = cross_val_predict(
            self.model, X, y=y, cv=self.cv_splitter, n_jobs=None, verbose=0, fit_params=self.fit_params,
            pre_dispatch='2*n_jobs', method='predict')

        yhat = pd.Series(yhat, index=y.index, name="yhat")
        yhat = self._cast_for_classifier_predict(yhat)

        eval_type: str = "cross_val"
        ret = cross_val_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)
        self.cross_val_predictions, self.cross_val_score_all, self.cross_val_score_split = ret
        self.cross_val_predictions.to_parquet(os.path.join(self.working_dir, f"{eval_type}_predictions.parquet"))
        self.cross_val_score_split.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_split.parquet"))
        self.cross_val_score_all.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_all.parquet"))

        ret = cross_val_data.evaluate(yhat=yhat, scoring_func=self.scoring_func, feature_neutral=True)
        _, self.cross_val_fnc_all, self.cross_val_fnc_split = ret
        self.cross_val_fnc_split.to_parquet(os.path.join(self.working_dir, "cross_val_fnc_split.parquet"))
        self.cross_val_fnc_all.to_parquet(os.path.join(self.working_dir, "cross_val_fnc_all.parquet"))
        return self

    def _save_prediction(self, eval_type: str = "cross_val"):
        self.cross_val_predictions.to_parquet(
            os.path.join(self.working_dir, f"{eval_type}_predictions.parquet"))
        self.cross_val_score_split.to_parquet(os.path.join(self.working_dir, "cross_val_score_split.parquet"))
        return self

    def _do_model_fit(self, data_type: Optional[str] = None):
        if not data_type:
            data_type = "training"

        if self.is_fitted and self.refresh_level > self.refresh_level_criterion:
            logging.info(f"skip due to refresh level ({self.refresh_level}) higher than {self.refresh_level_criterion}")
            return self

        logging.info(f"training model...")
        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_
        y = self._cast_for_classifier_fit(y)
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

        yhat = self._cast_for_classifier_predict(yhat)

        eval_type: str = "validation"
        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)
        valid_predictions, self.valid_score_all, self.valid_score_split = ret

        era = valid_data.groups_for_eval_
        if not self.data_manager.has_cast_mapping("index2label"):
            logging.info(f"rank predictions from regression model")
            valid_predictions["prediction"] = valid_predictions.groupby(
                era.name)["yhat"].apply(lambda x: scale_uniform(x))
        self.valid_predictions = valid_predictions

        self.valid_predictions.to_parquet(os.path.join(self.working_dir, f"{eval_type}_predictions.parquet"))
        self.valid_score_split.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_split.parquet"))
        self.valid_score_all.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_all.parquet"))

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
        era = infer_data.groups_for_eval_

        ret = pd.DataFrame(
            {"yhat": self.model.predict(X), "id": y.index, era.name: era.values, y.name: y.values})

        ret["yhat"] = self._cast_for_classifier_predict(ret["yhat"])
        if not self.data_manager.has_cast_mapping("index2label"):
            logging.info(f"rank predictions from regression model")
            ret["prediction"] = ret.groupby(era.name)["yhat"].apply(lambda x: scale_uniform(x))

        ret.to_parquet(os.path.join(self.working_dir, f"{data_type}_predictions.parquet"))
        ret.reindex(columns=["id", "prediction"]).to_csv(
            os.path.join(self.working_dir, f"{data_type}_predictions.csv"), index=False)
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
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, configs.unfitted_model_, scoring_func=configs.scoring_func,
            cv_splitter=configs.cv_splitter_, working_dir=output_data_path, **kwargs)


class EnsembleSolution(BaseSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", ensemble_method: str,
            solution_dirs: List[str], scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None,
            **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)
        self.ensemble_method: str = ensemble_method
        self.solution_dirs: List[str] = solution_dirs

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _ensemble_predictions(self, eval_type: str, groupby_col: str) -> pd.DataFrame:
        logging.info(f"ensemble for {eval_type}")
        predictions = list()
        for solution_dir in self.solution_dirs:
            file_path = os.path.join(solution_dir, f"{eval_type}_predictions.parquet")
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                logging.info(f"prediction file does not exist: {file_path}")
                continue

            df = pd.read_parquet(file_path)
            df["yhat"] = df.groupby(groupby_col)["yhat"].apply(lambda x: scale_uniform(x))
            predictions.append(df)

        if not predictions:
            logging.warning(f"no predictions to form ensemble.")
            return pd.DataFrame()

        df = pd.concat(predictions, sort=False)
        df = self._check_prediction(df)
        ret = df.reset_index().groupby([groupby_col] + df.index.names).mean().reset_index(groupby_col)
        logging.info(f"Generated {ret.shape[0]} predictions from {df.shape[0]} samples")

        # analytics
        df_corr = df.copy()
        cols = list()
        for i, pred in enumerate(predictions):
            col = f"yhat_{i:03d}"
            df_corr[col] = self._check_prediction(pred)["yhat"]
            cols.append(col)

        logging.info(
            f"correlation intra predictions:\n{df_corr.groupby(groupby_col)[cols].corr(method='spearman')}")
        return ret

    def _do_validation(self, data_type: Optional[str] = None):
        if data_type == "skip":
            return self

        if not data_type:
            data_type = "validation"

        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        eval_type: str = "validation"

        era = valid_data.groups_for_eval_

        ret = self._ensemble_predictions(eval_type, groupby_col=era.name)
        if ret.empty:
            return self

        ret["yhat"] = ret.groupby(era.name)["yhat"].apply(lambda x: scale_uniform(x))
        yhat = ret["yhat"]
        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)

        self.valid_predictions, self.valid_score_all, self.valid_score_split = ret
        self.valid_predictions.to_parquet(os.path.join(self.working_dir, f"{eval_type}_predictions.parquet"))
        self.valid_score_split.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_split.parquet"))
        self.valid_score_all.to_parquet(os.path.join(self.working_dir, f"{eval_type}_score_all.parquet"))

        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func, feature_neutral=True)
        _, self.valid_fnc_all, self.valid_fnc_split = ret
        self.valid_fnc_all.to_parquet(os.path.join(self.working_dir, "validation_fnc_all.parquet"))
        self.valid_fnc_split.to_parquet(os.path.join(self.working_dir, "validation_fnc_split.parquet"))

    def _do_inference(self, data_type: str = "tournament", ):
        logging.info(f"inference on {data_type}")

        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = infer_data.data_
        era = infer_data.groups_for_eval_

        ret = self._ensemble_predictions(data_type, groupby_col=era.name).reindex(index=y.index).reset_index()
        if ret.empty:
            return self

        if not self.data_manager.has_cast_mapping("index2label"):
            logging.info(f"rank predictions from regression model")
            ret["prediction"] = ret.groupby(era.name)["yhat"].apply(lambda x: scale_uniform(x))

        ret.to_parquet(os.path.join(self.working_dir, f"{data_type}_predictions.parquet"))
        ret.reindex(columns=["id", "prediction"]).to_csv(
            os.path.join(self.working_dir, f"{data_type}_predictions.csv"), index=False)
        return self

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._do_inference(data_type=infer_data_type)
        return self

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, configs.ensemble_method, configs.model_dirs,
            scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


if "__main__" == __name__:
    import pdb

    obj = Solution()
    pdb.set_trace()
