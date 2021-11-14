import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from typing import Optional, Callable, Any, Dict, List, Tuple
from scipy import stats

from ds_utils import FilenameTemplate as ft
from ds_utils.DefaultConfigs import RefreshLevel
from ds_utils.Solution.BaseSolution import MixinSolution
from ds_utils.Helper import INeutralizationHelper

_EPSILON: float = sys.float_info.min


def mean_predictions(data: pd.Series) -> float:
    return data.mean()


def geometric_mean_predictions(data: pd.Series) -> float:
    return stats.gmean(data)


_ENSEMBLE_METHODS: Dict[str, Callable] = {
    "mean": mean_predictions,
    "gmean": geometric_mean_predictions
}


class AdHocSolution(MixinSolution):
    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    @staticmethod
    def _check_prediction(df: pd.DataFrame) -> pd.DataFrame:
        if "id" in df.columns:
            df = df.set_index("id")

        return df

    def _process_predictions(self, eval_type: str) -> pd.DataFrame:
        raise NotImplementedError()

    def _read_predictions(
            self, solution_dir: str, eval_type: str = "validation", ):
        infer_data = self.data_manager.get_data_helper_by_type(data_type=eval_type, for_evaluation=False)
        file_path = os.path.join(solution_dir, ft.predictions_parquet_filename_template.format(eval_type=eval_type))
        if not (os.path.exists(file_path) and os.path.isfile(file_path)):
            logging.info(f"prediction file does not exist: {file_path}")
            return pd.DataFrame()

        return pd.read_parquet(file_path, columns=[infer_data.group_name_, self.default_yhat_pct_name])

    def _do_inference(self, data_type: str) -> pd.Series:
        predictions = self._process_predictions(data_type)
        if predictions.empty:
            logging.warning(f"skip inference since the processed predictions are empty")
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
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        raise NotImplementedError()


class EnsembleSolution(AdHocSolution):
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

    def _process_predictions(self, eval_type: str) -> pd.DataFrame:
        logging.info(f"ensemble for {eval_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=eval_type, for_evaluation=False)
        y = infer_data.y_
        groupby_col = infer_data.group_name_

        cols_prediction = list()
        df = pd.DataFrame()
        for i, solution_dir in enumerate(self.solution_dirs):
            df_prediction = self._read_predictions(eval_type=eval_type, solution_dir=solution_dir)
            if df_prediction.empty:
                continue

            col = f"{self.default_yhat_name}_{i:03d}"
            if df.empty:
                df = df_prediction
                df.rename(columns={self.default_yhat_pct_name: col}, inplace=True)
            else:
                df[col] = self._check_prediction(df_prediction)[self.default_yhat_pct_name]

            cols_prediction.append(col)
            logging.warning(f"Adding {col} ({df_prediction.shape}) into ensemble from {solution_dir}")

        if df.empty:
            logging.warning(f"no predictions to form ensemble predictions.")
            return pd.DataFrame()

        ret = df[[groupby_col]]
        ensemble_func = _ENSEMBLE_METHODS.get(self.ensemble_method, mean_predictions)
        ret[self.default_yhat_name] = df[cols_prediction].apply(lambda x: ensemble_func(x), axis=1)
        logging.info(f"Generated {ret.shape[0]} predictions from {df.shape[0]} samples")

        logging.info(f"correlation intra predictions:\n{df.groupby(groupby_col).corr(method='spearman')}")
        return ret.reindex(index=y.index)

    @classmethod
    def from_configs(
            cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        if configs.ensemble_method not in _ENSEMBLE_METHODS.keys():
            raise ValueError(f"no method to {configs.ensemble_method}")

        return cls(
            args.refresh_level, configs.data_manager_, ensemble_method=configs.ensemble_method,
            solution_dirs=configs.model_dirs, scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


class NeutralizeSolution(AdHocSolution):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", neutralization_gen: INeutralizationHelper,
            metric: str, pipeline_configs: Optional[List[Tuple[Any]]], quantiles: List[float],
            proportion_mapping: Dict[str, float], solution_dir: str, scoring_func: Optional[Callable] = None,
            working_dir: Optional[str] = None, **kwargs):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, scoring_func=scoring_func, working_dir=working_dir)
        self.solution_dir: str = solution_dir
        self.is_fitted: bool = True

        self.neutralization_gen: INeutralizationHelper = neutralization_gen
        self.metric: str = metric
        self.pipeline_configs: Optional[List[Tuple[Any]]] = pipeline_configs
        self.quantiles: List[float] = quantiles
        self.proportion_mapping: Dict[str, float] = proportion_mapping

    def _do_cross_val(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _do_model_fit(self, data_type: Optional[str] = None):
        raise NotImplementedError()

    def _process_predictions(self, eval_type: str) -> pd.DataFrame:
        logging.info(f"neutralize for {eval_type}")
        infer_data = self.data_manager.get_data_helper_by_type(data_type=eval_type, for_evaluation=True)
        y = infer_data.y_

        reference_filepath: str = os.path.join(
            ft.default_data_dir,
            ft.feature_analysis_filename_template.format(eval_type="training", target=infer_data.y_name_))
        reference = pd.read_parquet(reference_filepath)[self.metric]

        df_prediction = self._read_predictions(eval_type=eval_type, solution_dir=self.solution_dir)
        yhat = df_prediction[self.default_yhat_pct_name]
        neutralizer = self.neutralization_gen.from_arguments(
            reference, quantiles=self.quantiles, proportion_mapping=self.proportion_mapping,
            pipeline_configs=self.pipeline_configs)

        yhat = infer_data.neutralize_yhat_by_feature(yhat, neutralize_func=neutralizer.neutralize)
        return yhat.reindex(index=y.index).to_frame(self.default_yhat_name)

    @classmethod
    def from_configs(cls, args: Namespace, configs: "SolutionConfigs", output_data_path: str, **kwargs):
        return cls(
            args.refresh_level, configs.data_manager_, neutralization_gen=configs.neutralization_gen,
            metric=configs.metric, pipeline_configs=configs.pipeline_configs, quantiles=configs.quantiles,
            proportion_mapping=configs.proportion_mapping, solution_dir=configs.model_dir,
            scoring_func=configs.scoring_func, working_dir=output_data_path, **kwargs)


if "__main__" == __name__:
    pass
