import sys
import os
import json
import joblib
import logging
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
from ds_utils.DefaultConfigs import RefreshLevel

_EPSILON: float = sys.float_info.min


class BaseSolution:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):
        # default variable
        self.default_yhat_name: str = "yhat"
        self.default_yhat_pct_name: str = "prediction"

        self.is_fitted: bool = False

        self.refresh_level_criterion: RefreshLevel = RefreshLevel("model")
        self.refresh_level: RefreshLevel = refresh_level

        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.data_manager: "DataManager" = data_manager

        self.scoring_func: Callable = scoring_func
        self.feature_scoring_func: Optional[Callable] = None

    def _cast_for_classifier(self, target: pd.Series, cast_type: str):
        if self.data_manager.has_cast_mapping(cast_type):
            return self.data_manager.cast_target(target, cast_type)

        return target

    def _cast_for_classifier_fit(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="label2index")

    def _cast_for_classifier_predict(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="index2label")

    def _create_submission_file(self, yhat: pd.Series, data_type: str):
        filepath: str = os.path.join(self.working_dir, ft.predictions_csv_filename_template.format(eval_type=data_type))
        predictions = self.data_manager.get_example_data_by_type(data_type=data_type).squeeze()
        yhat.index.name = predictions.index.name
        yhat.reindex(index=predictions.index).rename(predictions.name).to_csv(filepath)

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit_for_training(self, data_type: str):
        raise NotImplementedError()

    def _do_model_fit(self, data_type: Optional[str] = None):
        if not data_type:
            data_type = "training"

        if self.is_fitted:
            return self

        self._do_model_fit_for_training(data_type=data_type)
        return self

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
        self._create_submission_file(yhat, data_type=data_type)
        return self

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        raise NotImplementedError()

    def _do_inference(self, data_type: str = "tournament", ):
        logging.info(f"inference on {data_type}")
        if not self.is_fitted:
            logging.warning(f"skip inference since model is not fitted yet")
            return self

        yhat = self._do_inference_for_tournament(data_type=data_type)
        self._create_submission_file(yhat, data_type=data_type)
        return self

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

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

    def _do_evaluation_on_learning_target(self, yhat: pd.Series, data_type: str = "validation") -> pd.Series:
        logging.info(f"evaluation on learning target: data_type={data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        results = valid_data.evaluate_with_y(yhat=yhat, scoring_func=self.scoring_func)

        score_all = results[1]
        score_all.to_parquet(
            os.path.join(self.working_dir, ft.score_target_all_filename_template.format(
                eval_type=data_type, target=valid_data.y_name_)))

        score_split = results[2]
        score_split.to_parquet(
            os.path.join(self.working_dir, ft.score_target_split_filename_template.format(
                eval_type=data_type, target=valid_data.y_name_)))

        predictions = results[0]
        return predictions

    def _do_evaluation_on_tournament_target(self, yhat: pd.Series, data_type: str = "validation") -> pd.Series:
        logging.info(f"evaluation on tournament target: data_type={data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)

        # feature exposure
        feature_evaluation = valid_data.evaluate_with_feature(yhat)
        feature_evaluation.to_parquet(
            os.path.join(self.working_dir, ft.feature_exposure_filename_template.format(eval_type=data_type)))

        # prediction results
        results = valid_data.evaluate_with_y(yhat=yhat, scoring_func=self.scoring_func)

        score_all = results[1]
        score_all.to_parquet(
            os.path.join(self.working_dir, ft.score_target_all_filename_template.format(
                eval_type=data_type, target=valid_data.y_name_)))

        score_split = results[2]
        score_split.to_parquet(
            os.path.join(self.working_dir, ft.score_target_split_filename_template.format(
                eval_type=data_type, target=valid_data.y_name_)))

        predictions = results[0]
        predictions[self.default_yhat_pct_name] = valid_data.pct_rank_normalize(predictions[self.default_yhat_name])
        predictions.to_parquet(
            os.path.join(self.working_dir, ft.predictions_parquet_filename_template.format(eval_type=data_type)))
        return predictions

    def _do_evaluation_on_tournament_example(self, yhat: pd.Series, data_type: str = "validation") -> pd.DataFrame:
        logging.info(f"evaluation on tournament example: data_type={data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)

        # corr with y example
        example_predictions = self.data_manager.get_example_data_by_type(data_type=data_type).squeeze()
        if not example_predictions.empty:
            example_analytics = valid_data.evaluate_with_y_and_example(yhat=yhat, example_yhat=example_predictions)
            example_analytics.to_parquet(
                os.path.join(self.working_dir, ft.example_analytics_filename_template.format(eval_type=data_type)))
            return example_analytics

        return pd.DataFrame()

    def _do_evaluation_on_feature_neutralize(self, yhat: pd.Series, data_type: str = "validation") -> pd.Series:
        logging.info(f"evaluation on feature neutralize: data_type={data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)

        # feature exposure
        feature_neutralize = valid_data.evaluate_with_feature_neutralize(
            yhat, cols_feature=None, proportion=1., normalize=True)
        feature_neutralize.to_frame("featureNeutralCorr").to_parquet(
            os.path.join(self.working_dir, ft.example_corr_filename_template.format(eval_type=data_type)))

        return feature_neutralize

    def _post_process_inference(self, yhat: pd.Series, data_type: str = "tournament") -> pd.Series:
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        predictions = yhat.to_frame()
        predictions[self.default_yhat_pct_name] = infer_data.pct_rank_normalize(predictions[yhat.name])
        predictions[infer_data.group_name_] = infer_data.groups_

        predictions.to_parquet(
            os.path.join(self.working_dir, ft.predictions_parquet_filename_template.format(eval_type=data_type)))
        return predictions

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        raise NotImplementedError()


class Solution(BaseSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", model: BaseEstimator,
            fit_params: Optional[Dict[str, Any]] = None, cv_splitter: Optional[BaseCrossValidator] = None,
            scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):

        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)

        self.model: Optional[BaseEstimator] = model
        self.fit_params: Optional[Dict[str, Any]] = fit_params if fit_params is not None else dict()

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter
        self.has_cv_splitter: bool = False
        if cv_splitter is not None:
            self.has_cv_splitter = True

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

        if self.is_fitted:
            logging.info(f"model fitted and loaded.")
            return self

        if self.refresh_level <= self.refresh_level_criterion:
            logging.info(
                f"skip loading model and refit ({self.refresh_level} <= {self.refresh_level_criterion})")
            return self

        logging.info(f"load pre-fitted model from {self.model_filepath_}")
        model = joblib.load(self.model_filepath_)
        if not isinstance(model, type(self.model)):
            logging.info(f"model mismatched: from file ({type(model)}) != model ({type(self.model)})")
            raise ValueError(f"model mismatched: from file ({type(model)}) != model ({type(self.model)})")

        self.model = model
        self.is_fitted = True
        return self

    def _do_cross_val(self, data_type: Optional[str] = None):
        pass

    def _do_model_fit_for_training(self, data_type: str):
        train_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = train_data.data_

        logging.info(f"training model...")
        self.model.fit(X, y, **self.fit_params)  # TODO: add fit_params:
        self.is_fitted = True
        self.save_model()
        return self

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        logging.info(f"inference for validation: {data_type}")
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = valid_data.data_
        yhat = pd.Series(self.model.predict(X), index=y.index, name=self.default_yhat_name)

        # target for learning
        if valid_data.y_name_ != "target":
            _ = self._do_evaluation_on_learning_target(yhat=yhat, data_type=data_type)

        # _ = self._do_evaluation_on_feature_neutralize(yhat=yhat, data_type=data_type)
        _ = self._do_evaluation_on_tournament_example(yhat=yhat, data_type=data_type)
        valid_predictions = self._do_evaluation_on_tournament_target(yhat=yhat, data_type=data_type)
        return valid_predictions[self.default_yhat_pct_name]

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        logging.info(f"inference for tournament: {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        X, y, groups = infer_data.data_

        predictions = y.to_frame()
        predictions[self.default_yhat_name] = self.model.predict(X)
        predictions = self._post_process_inference(yhat=predictions[self.default_yhat_name], data_type=data_type)
        return predictions[self.default_yhat_pct_name]

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
        self.is_fitted: bool = True

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit_for_training(self, data_type: str):
        raise NotImplementedError()

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _ensemble_predictions(self, eval_type: str, groupby_col: str) -> pd.DataFrame:
        data_helper = self.data_manager.get_data_helper_by_type(data_type=eval_type)
        logging.info(f"ensemble for {eval_type}")
        predictions = list()
        for solution_dir in self.solution_dirs:
            file_path = os.path.join(solution_dir, f"{eval_type}_predictions.parquet")
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                logging.info(f"prediction file does not exist: {file_path}")
                continue

            df_prediction = pd.read_parquet(file_path)
            if self.default_yhat_pct_name in df_prediction.columns:
                df_prediction[self.default_yhat_name] = df_prediction[self.default_yhat_pct_name]
            else:
                df_prediction[self.default_yhat_name] = data_helper.pct_rank_normalize(
                    df_prediction[self.default_yhat_name])

            predictions.append(df_prediction)

        if not predictions:
            logging.warning(f"no predictions to form ensemble predictions.")
            return pd.DataFrame()

        df = pd.concat(predictions, sort=False)
        df = self._check_prediction(df)

        ret = df.reset_index().groupby([groupby_col] + df.index.names)
        if self.ensemble_method == "mean":
            ret = ret.mean().reset_index(groupby_col)
        elif self.ensemble_method == "gmean":
            ret = ret[self.default_yhat_name].apply(stats.gmean).reset_index(groupby_col)

        logging.info(f"Generated {ret.shape[0]} predictions from {df.shape[0]} samples")

        # analytics
        df_corr = df.copy()
        cols = list()
        for i, prediction in enumerate(predictions):
            col = f"{self.default_yhat_name}_{i:03d}"
            df_corr[col] = self._check_prediction(prediction)[self.default_yhat_name]
            cols.append(col)

        logging.info(
            f"correlation intra predictions:\n{df_corr.groupby(groupby_col)[cols].corr(method='spearman')}")
        return ret

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        y = valid_data.y_

        predictions = self._ensemble_predictions(data_type, groupby_col=valid_data.group_name_).reindex(index=y.index)
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        yhat = predictions[self.default_yhat_name]
        # _ = self._do_evaluation_on_feature_neutralize(yhat=yhat, data_type=data_type)
        _ = self._do_evaluation_on_tournament_example(yhat=yhat, data_type=data_type)
        valid_predictions = self._do_evaluation_on_tournament_target(yhat=yhat, data_type=data_type)
        return valid_predictions[self.default_yhat_pct_name]

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        y = infer_data.y_

        predictions = self._ensemble_predictions(
            data_type, groupby_col=infer_data.group_name_).reindex(index=y.index)
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        predictions = self._post_process_inference(yhat=predictions[self.default_yhat_pct_name], data_type=data_type)
        return predictions[self.default_yhat_pct_name]

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._do_inference(data_type=infer_data_type)
        return self

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    @classmethod
    def from_configs(
            cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, ensemble_method=configs.ensemble_method,
            solution_dirs=configs.model_dirs, scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


class NeutralizeSolution(BaseSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager",
            scoring_func: Optional[Callable] = None, working_dir: Optional[str] = None,
            **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit_for_training(self, data_type: str):
        raise NotImplementedError()

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=False)
        predictions = pd.DataFrame()  # TODO: Neutralize
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        valid_predictions = self._do_evaluation_on_tournament_target(
            yhat=predictions[self.default_yhat_name], data_type=data_type)
        return valid_predictions[self.default_yhat_pct_name]

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        logging.info(f"inference on {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type, for_evaluation=True)
        X, y, groups = infer_data.data_

        predictions = self._ensemble_predictions(
            data_type, groupby_col=infer_data.group_name_).reindex(index=y.index).reset_index()
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        predictions = self._post_process_inference(yhat=predictions[self.default_yhat_name], data_type=data_type)
        return predictions[self.default_yhat_pct_name]

    def evaluate(self, train_data_type: str = "training", valid_data_type: str = "validation", ):
        self._do_validation(data_type=valid_data_type)
        return self

    def run(self, train_data_type: str = "training", infer_data_type: str = "tournament", ):
        self._do_inference(data_type=infer_data_type)
        return self

    def save_model(self):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, configs.ensemble_method, configs.model_dirs,
            scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


if "__main__" == __name__:
    import pdb

    obj = Solution()
    pdb.set_trace()
