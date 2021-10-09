import sys
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from typing import Optional, Callable, Any, Dict, List, Tuple
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

from ds_utils.DefaultConfigs import RefreshLevel
from ds_utils.Solution.BaseSolution import MixinSolution

_EPSILON: float = sys.float_info.min


class ScikitEstimatorSolution(MixinSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None, **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)

        self.model: Optional[BaseEstimator] = model
        self.fit_params: Optional[Dict[str, Any]] = fit_params if fit_params is not None else dict()

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter
        self.has_cv_splitter: bool = False
        if cv_splitter is not None:
            self.has_cv_splitter = True

    def _cast_for_classifier(self, target: pd.Series, cast_type: str):
        if self.data_manager.has_cast_mapping(cast_type):
            return self.data_manager.cast_target(target, cast_type)

        return target

    def _cast_for_classifier_fit(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="label2index")

    def _cast_for_classifier_predict(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="index2label")

    @property
    def model_filepath_(self) -> str:
        return os.path.join(self.working_dir, "model.pkl")

    def _save_model(self):
        if not self.is_fitted:
            logging.info(f"cannot save model due to not fitted")
            return self

        logging.info(f"save model to {self.model_filepath_}")
        joblib.dump(deepcopy(self.model), self.model_filepath_)
        return self

    def _load_model(self):
        _model_filepath = self.model_filepath_
        if not (os.path.exists(_model_filepath) or os.path.isfile(_model_filepath)):
            return self

        if self.is_fitted:
            logging.info(f"model fitted and loaded, from {_model_filepath}")
            return self

        if self.refresh_level <= self.refresh_level_criterion:
            logging.info(
                f"skip loading model and refit ({self.refresh_level} <= {self.refresh_level_criterion})")
            return self

        logging.info(f"load pre-fitted model from {_model_filepath}")
        model = joblib.load(_model_filepath)
        if not isinstance(model, type(self.model)):
            logging.info(f"model mismatched: from file ({type(model)}) != model ({type(self.model)})")
            raise ValueError(f"model mismatched: from file ({type(model)}) != model ({type(self.model)})")

        self.model = model
        self.is_fitted = True
        return self

    def _do_cross_val(self, data_type: Optional[str] = None):
        pass

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        raise NotImplementedError()

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        raise NotImplementedError()

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_cross_val(data_type=train_data_type)

        self._load_model()
        self._do_model_fit(data_type=train_data_type)
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._load_model()
        self._do_model_fit(data_type=train_data_type)
        self._do_tournament(data_type=infer_data_type)
        return self

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            refresh_level=args.refresh_level, data_manager=configs.data_manager_, model=configs.unfitted_model_,
            scoring_func=configs.scoring_func, cv_splitter=configs.cv_splitter_, working_dir=output_data_path, **kwargs)


class RegressorSolution(ScikitEstimatorSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None, **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, model=model, fit_params=fit_params,
            cv_splitter=cv_splitter, scoring_func=scoring_func, working_dir=working_dir, *kwargs)

    def _do_model_fit(self, data_type: Optional[str] = None):
        if self.is_fitted:
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_

        logging.info(f"training model...")
        self.model.fit(X, y, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self._save_model()
        return self

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        logging.info(f"inference for validation: {data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = valid_data.data_
        return pd.Series(self.model.predict(X), index=y.index, name=self.default_yhat_name)

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        logging.info(f"inference for tournament: {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = infer_data.data_

        predictions = y.to_frame()
        predictions[self.default_yhat_name] = self.model.predict(X)
        return predictions[self.default_yhat_name]


class RankerSolution(ScikitEstimatorSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None, **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, model=model, fit_params=fit_params,
            cv_splitter=cv_splitter, scoring_func=scoring_func, working_dir=working_dir, **kwargs)

    def _do_model_fit(self, data_type: Optional[str] = None):
        if self.is_fitted:
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_

        _groups = train_data.group_counts_

        logging.info(f"training model...")
        self.model.fit(X, self._cast_for_classifier_fit(y), group=_groups, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self._save_model()
        return self

    def inference_on_group(self, infer_data) -> pd.Series:
        X, y, groups = infer_data.data_
        X[groups.name] = groups

        yhat = X.groupby(groups.name).apply(lambda x: pd.Series(
            self.model.predict(x[infer_data.cols_feature]), index=x.index)).rename(self.default_yhat_name)

        return yhat.reset_index(groups.name)[self.default_yhat_name]

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        logging.info(f"inference for validation: {data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        return self.inference_on_group(infer_data=valid_data)

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        logging.info(f"inference for tournament: {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)

        predictions = infer_data.y_.to_frame()
        predictions[self.default_yhat_name] = self.inference_on_group(infer_data=infer_data)
        return predictions[self.default_yhat_name]


class CatBoostRankerSolution(RankerSolution):
    def _do_model_fit(self, data_type: Optional[str] = None):
        if self.is_fitted:
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_

        logging.info(f"training model...")
        self.model.fit(X, y, group_id=groups, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self._save_model()
        return self


class ClassifierSolution(ScikitEstimatorSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None, **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, model=model, fit_params=fit_params,
            cv_splitter=cv_splitter, scoring_func=scoring_func, working_dir=working_dir, **kwargs)

    def _do_model_fit(self, data_type: Optional[str] = None):
        if self.is_fitted:
            return self

        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_

        logging.info(f"training model...")
        self.model.fit(X, self._cast_for_classifier_fit(y), **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self._save_model()
        return self

    def _cast_for_classifier_predict_proba(self, target: np.ndarray) -> np.ndarray:
        return np.dot(target, list(self.data_manager.index2label.values()))

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        logging.info(f"inference for validation: {data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = valid_data.data_

        yhat = self.model.predict_proba(X)
        # yhat = self._cast_for_classifier_predict(pd.Series(yhat, index=y.index, name=self.default_yhat_name))
        return pd.Series(
            self._cast_for_classifier_predict_proba(yhat), index=y.index, name=self.default_yhat_name)

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        logging.info(f"inference for tournament: {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = infer_data.data_

        predictions = y.to_frame()
        # predictions[self.default_yhat_name] = self.model.predict(X)
        # predictions[self.default_yhat_name] = self._cast_for_classifier_predict(predictions[self.default_yhat_name])
        predictions[self.default_yhat_name] = self._cast_for_classifier_predict_proba(self.model.predict_proba(X))
        return predictions[self.default_yhat_name]


if "__main__" == __name__:
    pass
