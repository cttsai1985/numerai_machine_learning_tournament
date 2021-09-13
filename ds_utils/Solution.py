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
from sklearn.base import BaseEstimator

from .DefaultConfigs import RefreshLevel

_scoreSplitFilename: str = "{eval_type}_score_split.parquet"
_scoreAllFilename: str = "{eval_type}_score_all.parquet"
_scoreTargetSplitFilename: str = "{eval_type}_score_split_{target}.parquet"
_scoreTargetAllFilename: str = "{eval_type}_score_all_{target}.parquet"
_predictionsParquetFilename: str = "{eval_type}_predictions.parquet"
_predictionsCsvFilename: str = "{eval_type}_predictions.csv"


class BaseSolution:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):
        self.refresh_level: RefreshLevel = refresh_level
        self.refresh_level_criterion: RefreshLevel = RefreshLevel("model")

        self.is_fitted: bool = False

        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        self.data_manager: "DataManager" = data_manager

        self.scoring_func: Callable = scoring_func

    def _cast_for_classifier(self, target: pd.Series, cast_type: str):
        if self.data_manager.has_cast_mapping(cast_type):
            return self.data_manager.cast_target(target, cast_type)

        return target

    def _cast_for_classifier_fit(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="label2index")

    def _cast_for_classifier_predict(self, target: pd.Series) -> pd.Series:
        return self._cast_for_classifier(target=target, cast_type="index2label")

    def _create_submission_file(self, yhat: pd.Series, data_type: str):
        filepath: str = os.path.join(self.working_dir, _predictionsCsvFilename.format(eval_type=data_type))
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
        logging.info(f"inference on {data_type}")
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
        yhat_name: str = "yhat"
        yhat_pct_name: str = "prediction"

        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = valid_data.data_
        yhat = pd.Series(self.model.predict(X), index=y.index, name=yhat_name)

        # target for learning
        if valid_data.y_name_ != "target":
            ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func, eval_training_target=True)
            ret[1].to_parquet(os.path.join(
                self.working_dir, _scoreTargetAllFilename.format(eval_type=data_type, target=valid_data.y_name_)))
            ret[2].to_parquet(os.path.join(
                self.working_dir, _scoreTargetSplitFilename.format(eval_type=data_type, target=valid_data.y_name_)))

        # target for eval
        ret = valid_data.evaluate(yhat=yhat, scoring_func=self.scoring_func)
        ret[1].to_parquet(os.path.join(self.working_dir, _scoreAllFilename.format(eval_type=data_type)))
        ret[2].to_parquet(os.path.join(self.working_dir, _scoreSplitFilename.format(eval_type=data_type)))

        valid_predictions = ret[0]
        valid_predictions[yhat_pct_name] = valid_data.pct_rank_normalize(valid_predictions[yhat_name])

        valid_predictions.to_parquet(
            os.path.join(self.working_dir, _predictionsParquetFilename.format(eval_type=data_type)))
        return valid_predictions[yhat_pct_name]

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        yhat_name: str = "yhat"
        yhat_pct_name: str = "prediction"

        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = infer_data.data_

        predictions = y.to_frame()
        predictions[yhat_name] = self.model.predict(X)
        predictions[yhat_pct_name] = infer_data.pct_rank_normalize(predictions[yhat_name])
        predictions[infer_data.group_name_for_eval_] = infer_data.groups_for_eval_

        predictions.to_parquet(
            os.path.join(self.working_dir, _predictionsParquetFilename.format(eval_type=data_type)))
        return predictions[yhat_pct_name]

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
        raise NotImplementedError()

    def _do_model_fit_for_training(self, data_type: str):
        raise NotImplementedError()

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _ensemble_predictions(self, eval_type: str, groupby_col: str) -> pd.DataFrame:
        yhat_name: str = "yhat"
        yhat_pct_name: str = "prediction"

        data_helper = self.data_manager.get_data_helper_by_type(data_type=eval_type)
        logging.info(f"ensemble for {eval_type}")
        predictions = list()
        for solution_dir in self.solution_dirs:
            file_path = os.path.join(solution_dir, f"{eval_type}_predictions.parquet")
            if not (os.path.exists(file_path) and os.path.isfile(file_path)):
                logging.info(f"prediction file does not exist: {file_path}")
                continue

            df_prediction = pd.read_parquet(file_path)
            if yhat_pct_name in df_prediction.columns:
                df_prediction[yhat_name] = df_prediction[yhat_pct_name]
            else:
                df_prediction[yhat_name] = data_helper.pct_rank_normalize(df_prediction[yhat_name])

            predictions.append(df_prediction)

        if not predictions:
            logging.warning(f"no predictions to form ensemble predictions.")
            return pd.DataFrame()

        df = pd.concat(predictions, sort=False)
        df = self._check_prediction(df)
        ret = df.reset_index().groupby([groupby_col] + df.index.names).mean().reset_index(groupby_col)
        logging.info(f"Generated {ret.shape[0]} predictions from {df.shape[0]} samples")

        # analytics
        df_corr = df.copy()
        cols = list()
        for i, prediction in enumerate(predictions):
            col = f"{yhat_name}_{i:03d}"
            df_corr[col] = self._check_prediction(prediction)[yhat_name]
            cols.append(col)

        logging.info(
            f"correlation intra predictions:\n{df_corr.groupby(groupby_col)[cols].corr(method='spearman')}")
        return ret

    def _do_inference_for_validation(self, data_type: str) -> pd.Series:
        yhat_name: str = "yhat"
        yhat_pct_name: str = "prediction"

        valid_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        predictions = self._ensemble_predictions(data_type, groupby_col=valid_data.group_name_for_eval_)
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        ret = valid_data.evaluate(yhat=predictions[yhat_name], scoring_func=self.scoring_func)
        ret[1].to_parquet(os.path.join(self.working_dir, _scoreAllFilename.format(eval_type=data_type)))
        ret[2].to_parquet(os.path.join(self.working_dir, _scoreSplitFilename.format(eval_type=data_type)))

        valid_predictions = ret[0]
        valid_predictions[yhat_pct_name] = valid_data.pct_rank_normalize(valid_predictions[yhat_name])

        valid_predictions.to_parquet(
            os.path.join(self.working_dir, _predictionsParquetFilename.format(eval_type=data_type)))
        return valid_predictions[yhat_pct_name]

    def _do_inference_for_tournament(self, data_type: str) -> pd.Series:
        yhat_name: str = "yhat"
        yhat_pct_name: str = "prediction"

        logging.info(f"inference on {data_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=data_type)
        X, y, groups = infer_data.data_

        predictions = self._ensemble_predictions(
            data_type, groupby_col=infer_data.group_name_for_eval_).reindex(index=y.index).reset_index()
        if predictions.empty:
            logging.warning(f"skip inference since the ensemble predictions are empty")
            return self

        predictions[yhat_pct_name] = infer_data.pct_rank_normalize(predictions[yhat_name])
        predictions.to_parquet(
            os.path.join(self.working_dir, _predictionsParquetFilename.format(eval_type=data_type)))
        return predictions[yhat_pct_name]

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
