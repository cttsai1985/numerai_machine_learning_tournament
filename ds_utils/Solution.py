import sys
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import BaseCrossValidator
from scipy import stats
from sklearn.base import BaseEstimator

from ds_utils import FilenameTemplate as ft
from ds_utils import Utils
from ds_utils.DefaultConfigs import RefreshLevel

_EPSILON: float = sys.float_info.min


class ISolution:
    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        raise NotImplementedError()

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        raise NotImplementedError()

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        raise NotImplementedError()


class MixinSolution(ISolution):
    """
    Useful Mixin Class for lots of utility
    """

    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):
        # default variable
        self.default_yhat_name: str = "yhat"
        self.default_yhat_pct_name: str = "prediction"

        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.data_manager: "DataManager" = data_manager

        self.scoring_func: Callable = scoring_func
        self.feature_scoring_func: Optional[Callable] = None

        # default state
        self.is_fitted: bool = False

        self.refresh_level_criterion: RefreshLevel = RefreshLevel("model")
        self.refresh_level: RefreshLevel = refresh_level

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        raise NotImplementedError()

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        raise NotImplementedError()

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        raise NotImplementedError()

    def _do_evaluation_on_feature_neutralize(
            self, yhat: pd.Series, data_type: str = "validation", tb_num: Optional[int] = None) -> pd.Series:
        """

        :param yhat:
        :param data_type:
        :return:
        """
        feature_neutral_filepath = os.path.join(
            self.working_dir, ft.feature_neutral_filename_template.format(eval_type=data_type))
        if tb_num:
            logging.info(f"evaluation on feature neutralize: data_type={data_type}, tb_num={tb_num:04d}")
            feature_neutral_filepath = os.path.join(
                self.working_dir, ft.feature_neutral_tb_filename_template.format(eval_type=data_type, tb_num=tb_num))
        else:
            logging.info(f"evaluation on feature neutralize: data_type={data_type}")

        # feature exposure
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        feature_neutralize = valid_data.evaluate_with_feature_neutralize(
            yhat, cols_feature=None, proportion=1., normalize=True)
        feature_neutralize.to_frame("featureNeutralCorr").to_parquet(feature_neutral_filepath)
        return feature_neutralize

    def _do_evaluation_on_example_prediction(
            self, yhat: pd.Series, data_type: str = "validation", tb_num: Optional[int] = None) -> pd.DataFrame:
        """
        Example Analytics such as Corr with Example Predictions and Meta Model Control
        :param yhat:
        :param data_type:
        :return:
        """
        example_predictions = self.data_manager.get_example_data_by_type(data_type=data_type).squeeze()
        if example_predictions.empty:
            return pd.DataFrame()

        example_analytics_filepath = os.path.join(
            self.working_dir, ft.example_analytics_filename_template.format(eval_type=data_type))
        if not tb_num:
            logging.info(f"evaluation on tournament example: data_type={data_type}")
        else:
            logging.info(f"evaluation on tournament example: data_type={data_type}, tb_num={tb_num:04d}")
            example_analytics_filepath = os.path.join(
                self.working_dir, ft.example_analytics_tb_filename_template.format(eval_type=data_type, tb_num=tb_num))

        # corr with y example
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        example_analytics = valid_data.evaluate_with_y_and_example(yhat=yhat, example_yhat=example_predictions)
        example_analytics.to_parquet(example_analytics_filepath)
        return example_analytics

    def _do_evaluation_on_learning_target(self, yhat: pd.Series, data_type: str = "validation") -> pd.Series:
        """

        :param yhat:
        :param data_type:
        :return:
        """
        logging.info(f"evaluation on learning target: data_type={data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        results = valid_data.evaluate_with_y(yhat=yhat, scoring_func=self.scoring_func)
        y_name_ = valid_data.y_name_

        score_target_all_filepath = os.path.join(
            self.working_dir, ft.score_target_all_filename_template.format(eval_type=data_type, target=y_name_))
        score_target_split_filepath = os.path.join(
            self.working_dir, ft.score_target_split_filename_template.format(eval_type=data_type, target=y_name_))

        score_all = results[1]
        score_all.to_parquet(score_target_all_filepath)

        score_split = results[2]
        score_split.to_parquet(score_target_split_filepath)

        predictions = results[0]
        return predictions

    def _do_evaluation_on_tournament_target(
            self, yhat: pd.Series, data_type: str = "validation", tb_num: Optional[int] = None) -> pd.Series:
        """

        :param yhat:
        :param data_type:
        :return:
        """
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)

        predictions_parquet_filepath = os.path.join(
            self.working_dir, ft.predictions_parquet_filename_template.format(eval_type=data_type))
        if tb_num:
            logging.info(f"evaluation on tournament target: data_type={data_type}, tb_num={tb_num:04d}")
            feature_exposure_filepath = os.path.join(
                self.working_dir, ft.feature_exposure_tb_filename_template.format(eval_type=data_type, tb_num=tb_num))
            score_target_all_filepath = os.path.join(
                self.working_dir, ft.score_tb_all_filename_template.format(eval_type=data_type, tb_num=tb_num))
            score_target_split_filepath = os.path.join(
                self.working_dir, ft.score_tb_split_filename_template.format(eval_type=data_type, tb_num=tb_num))
        else:
            logging.info(f"evaluation on tournament target: data_type={data_type}")
            feature_exposure_filepath = os.path.join(
                self.working_dir, ft.feature_exposure_filename_template.format(eval_type=data_type))
            score_target_all_filepath = os.path.join(
                self.working_dir, ft.score_all_filename_template.format(eval_type=data_type))
            score_target_split_filepath = os.path.join(
                self.working_dir, ft.score_split_filename_template.format(eval_type=data_type))

        # feature exposure
        valid_data.evaluate_with_feature(yhat).to_parquet(feature_exposure_filepath)

        # performance evaluation for prediction results
        results = valid_data.evaluate_with_y(yhat=yhat, scoring_func=self.scoring_func)

        score_all = results[1]
        score_all.to_parquet(score_target_all_filepath)

        score_split = results[2]
        score_split.to_parquet(score_target_split_filepath)

        predictions = results[0]
        if tb_num:
            return predictions

        predictions[self.default_yhat_pct_name] = valid_data.pct_rank_normalize(predictions[self.default_yhat_name])
        predictions.to_parquet(predictions_parquet_filepath)
        return predictions

    def _post_process_tournament(self, yhat: pd.Series, data_type: str = "tournament") -> pd.Series:
        """
        
        :param yhat:
        :param data_type:
        :return:
        """
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        predictions_parquet_filepath = os.path.join(
            self.working_dir, ft.predictions_parquet_filename_template.format(eval_type=data_type))

        predictions = yhat.to_frame()
        predictions[self.default_yhat_pct_name] = infer_data.pct_rank_normalize(predictions[yhat.name])
        predictions[infer_data.group_name_] = infer_data.groups_

        predictions.to_parquet(predictions_parquet_filepath)
        return predictions

    def _create_submission_file(self, yhat: pd.Series, data_type: str = "validation") -> pd.Series:
        """

        :param yhat:
        :param data_type:
        :return:
        """
        filepath: str = os.path.join(self.working_dir, ft.predictions_csv_filename_template.format(eval_type=data_type))
        predictions = self.data_manager.get_example_data_by_type(data_type=data_type).squeeze()
        yhat.index.name = predictions.index.name
        yhat.reindex(index=predictions.index).rename(predictions.name).to_csv(filepath)
        return yhat

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        raise NotImplementedError()

    def _do_validation(self, data_type: Optional[str] = None):
        logging.info(f"validation on {data_type}")
        if data_type == "skip":
            return self

        if not data_type:
            data_type = "validation"

        if not self.is_fitted:
            logging.warning(f"skip model validation since model is not fitted yet")
            return self

        yhat = self._do_inference_for_validation(data_type=data_type)

        predictions = self._do_evaluation_on_tournament_target(yhat=yhat, data_type=data_type)
        self._create_submission_file(predictions[self.default_yhat_pct_name], data_type=data_type)

        # _ = self._do_evaluation_on_feature_neutralize(yhat=yhat, data_type=data_type)
        _ = self._do_evaluation_on_example_prediction(yhat=yhat, data_type=data_type)
        # target for learning
        if self.data_manager.col_target != "target":
            _ = self._do_evaluation_on_learning_target(yhat=yhat, data_type=data_type)

        # get top and bottom predictions
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        tb_num = 200
        groups = valid_data.groups_
        yhat_tb = Utils.select_series_from_tb_num(yhat, groups=groups, tb_num=tb_num)

        _ = self._do_evaluation_on_example_prediction(yhat=yhat_tb, data_type=data_type, tb_num=tb_num)
        _ = self._do_evaluation_on_tournament_target(yhat=yhat_tb, data_type=data_type, tb_num=tb_num)
        _ = self._do_evaluation_on_feature_neutralize(yhat=yhat_tb, data_type=data_type, tb_num=tb_num)
        return self

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        raise NotImplementedError()

    def _do_tournament(self, data_type: Optional[str] = None):
        logging.info(f"inference on {data_type}")
        if not self.is_fitted:
            logging.warning(f"skip inference since model is not fitted yet")
            return self

        yhat = self._do_inference_for_tournament(data_type=data_type)
        predictions = self._post_process_tournament(yhat=yhat, data_type=data_type)
        self._create_submission_file(predictions[self.default_yhat_pct_name], data_type=data_type)
        return self


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


class EnsembleSolution(MixinSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", ensemble_method: str,
            solution_dirs: List[str], scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None,
            **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)
        self.ensemble_method: str = ensemble_method
        self.solution_dirs: List[str] = solution_dirs
        self.is_fitted: bool = True

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _ensemble_predictions(self, eval_type: str) -> pd.DataFrame:
        logging.info(f"ensemble for {eval_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=eval_type, for_evaluation=False)
        y = infer_data.y_
        groupby_col = infer_data.group_name_

        cols_prediction = list()
        df = pd.DataFrame()
        for i, solution_dir in enumerate(self.solution_dirs):
            file_path = os.path.join(solution_dir, ft.predictions_parquet_filename_template.format(eval_type=eval_type))
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                logging.info(f"prediction file does not exist: {file_path}")
                continue

            col = f"{self.default_yhat_name}_{i:03d}"
            df_prediction = pd.read_parquet(file_path, columns=[groupby_col, self.default_yhat_pct_name])
            if df.empty:
                df = df_prediction
                df.rename(columns={self.default_yhat_pct_name: col}, inplace=True)
            else:
                df[col] = self._check_prediction(df_prediction)[self.default_yhat_pct_name]

            cols_prediction.append(col)
            logging.warning(f"Adding {col} into ensemble from {file_path}")

        if df.empty:
            logging.warning(f"no predictions to form ensemble predictions.")
            return pd.DataFrame()

        ret = df[[groupby_col]]
        if self.ensemble_method == "mean":
            ret[self.default_yhat_name] = df[cols_prediction].apply(lambda x: x.mean(), axis=1)
        elif self.ensemble_method == "gmean":
            ret[self.default_yhat_name] = df[cols_prediction].apply(lambda x: stats.gmean(x), axis=1)

        logging.info(f"Generated {ret.shape[0]} predictions from {df.shape[0]} samples")

        logging.info(f"correlation intra predictions:\n{df.groupby(groupby_col).corr(method='spearman')}")
        return ret.reindex(index=y.index)

    def _do_inference(self, data_type: str) -> pd.Series:
        predictions = self._ensemble_predictions(data_type)
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        return predictions[self.default_yhat_name]

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        return self._do_inference(data_type=data_type)

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        return self._do_inference(data_type=data_type)

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._do_tournament(data_type=infer_data_type)
        return self

    @classmethod
    def from_configs(
            cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, ensemble_method=configs.ensemble_method,
            solution_dirs=configs.model_dirs, scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


class AutoSolution(ISolution):
    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        raise NotImplementedError()

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        raise NotImplementedError()

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        if configs.model_gen_query == "CatBoostRanker":
            logging.info(f"RankerSolution")
            configs.data_manager_.configure_label_index_mapping()
            return CatBoostRankerSolution.from_configs(args=args, configs=configs, output_data_path=output_data_path)

        if configs.model_gen_query.endswith("Ranker"):
            logging.info(f"RankerSolution")
            configs.data_manager_.configure_label_index_mapping()
            return RankerSolution.from_configs(args=args, configs=configs, output_data_path=output_data_path)

        if configs.model_gen_query.endswith("Classifier"):
            logging.info(f"ClassifierSolution")
            configs.data_manager_.configure_label_index_mapping()
            return ClassifierSolution.from_configs(args=args, configs=configs, output_data_path=output_data_path)

        return RegressorSolution.from_configs(args=args, configs=configs, output_data_path=output_data_path)


if "__main__" == __name__:
    pass
