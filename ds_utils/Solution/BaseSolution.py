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


if "__main__" == __name__:
    pass
