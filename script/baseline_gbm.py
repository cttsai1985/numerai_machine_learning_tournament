import sys
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator

import lightgbm as lgb
from lightgbm import LGBMRegressor

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import PerformanceTracker, Solution, DataManager, OptunaLGBMTuner, Helper
from ds_utils import RefreshLevel


class SolutionConfigs:
    def __init__(self, eval_data_type: str = None, num_boost_round: int = 2000):
        self.working_dir: str = "../input/numerai_tournament_resource/baseline/lightgbm_optuna/"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        self.input_data_dir: str = "../input/numerai_tournament_resource/latest_tournament_datasets/"

        data_file_types: List[str] = ["training", "validation", "test", "live", "tournament"]
        data_filenames: List[str] = [
            "numerai_training_data.parquet", "numerai_validation_data.parquet", "numerai_test_data.parquet",
            "numerai_live_data.parquet", "numerai_tournament_data.parquet"]

        self.data_mapping: Optional[Dict[str, str]] = {k: v for k, v in zip(data_file_types, data_filenames)}
        self.eval_data_type: str = eval_data_type if eval_data_type is not None else "training"

        self.column_target: str = "target"
        self.columns_group: List[str] = ["era"]

        meta_data_path: str = "../input/numerai_tournament_resource/metadata/"
        feature_columns_set = os.path.join(meta_data_path, "features_numerai.json")
        with open(feature_columns_set, "r") as fp:
            columns_feature = json.load(fp)

        self.columns_feature: List[str] = columns_feature

        self.data_manager: Optional[DataManager] = None

        self.template_cv_splitter_gen: BaseCrossValidator = StratifiedKFold
        self.template_cv_splitter_params: Dict[str, Any] = {"n_splits": 5, "shuffle": True, "random_state": 42}
        self.cv_splitter: Optional[BaseCrossValidator] = None

        self.solution: Optional[Solution] = None
        self.model_gen: BaseEstimator = LGBMRegressor
        self.model_base_params: Dict[str, Any] = {
            "objective": "regression",  # "huber", "fair"
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": .01,
            "max_depth": 12,
            "n_jobs": 12
        }
        self.model_best_params: Optional[Dict[str, Any]] = None
        self.model_tuning_seed: int = 42
        self.num_boost_round: int = num_boost_round

        self.scoring_func: Callable = PerformanceTracker().score

    @property
    def solution_(self) -> Solution:
        if self.solution is None:
            logging.info(f"initialize Solution on {self.working_dir}")
            self.solution = Solution.from_configs(self)

        return self.solution

    @property
    def data_manager_(self) -> DataManager:
        if self.data_manager is None:
            logging.info(f"initialize data manager on {self.input_data_dir}")
            self.data_manager = DataManager.from_configs(self)

        return self.data_manager

    @property
    def unfitted_model_(self) -> BaseEstimator:
        if self.model_best_params is None:
            logging.info(f"generate model {self.model_gen} with base paramters: {self.base_params}")
            return self.model_gen(**self.base_params)

        logging.info(f"generate model {self.model_gen} with base paramters: {self.model_best_params}")
        return self.model_gen(**self.model_best_params)

    @property
    def template_cv_splitter_(self) -> BaseCrossValidator:
        return self.template_cv_splitter_gen(**self.template_cv_splitter_params)

    @property
    def cv_splitter_(self) -> BaseCrossValidator:
        if self.cv_splitter is None:
            self.cv_splitter = Helper.PredefinedSplitHelper.from_configs(
                configs=self).produce(data_type=self.eval_data_type)

        return self.cv_splitter

    @classmethod
    def from_config_file(cls, file_path: str):
        pass


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solver", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None, help="folder for data")
    parser.add_argument(
        "--eval-data-type", type=str, default="training", help="data for training and evaluation on cross validation")

    parser.add_argument(
        "--refresh-level", type=str, default=None,
        help=f"determine the level of computing: {RefreshLevel.possible_refresh_levels()}")
    parser.add_argument("--compute-hpo", action="store_true", help="run hyper-parameter tuning")
    parser.add_argument("--compute-eval", action="store_true", help="run cross validation and validation")
    parser.add_argument("--compute-infer", action="store_true", help="run inference")
    parser.add_argument("--compute-all", action="store_true", help="run all compute")

    args = parser.parse_args()
    args.refresh_level = RefreshLevel(args.refresh_level)
    if args.compute_all:
        args.compute_hpo = True
        args.compute_eval = True
        args.compute_infer = True
    return args


def main(args: argparse.Namespace):
    output_data_path: str = "../input/numerai_tournament_resource/baseline/lightgbm_optuna_foobar/"

    configs = SolutionConfigs(eval_data_type=args.eval_data_type, num_boost_round=5)
    tuner = OptunaLGBMTuner.from_configs(
        args=args, configs=configs, output_data_path=output_data_path).load_optimized_params()
    if args.compute_hpo and args.refresh_level <= RefreshLevel("hyper_parameters"):
        tuner.run(data_type=args.eval_data_type)

    configs.model_best_params = tuner.params_
    solver = Solution.from_configs(args=args, configs=configs, output_data_path=output_data_path)
    if args.compute_eval:
        solver.evaluate(train_data_type=args.eval_data_type, valid_data_type="validation")

    if args.compute_infer:
        solver.run()

    return


if "__main__" == __name__:
    ds_utils.initialize_logger()
    _args = parse_commandline()
    main(_args)

    # TODO: model helper
    # TODO: parameterize script
    # TODO: model analytics
    # TODO: Post processing
    # TODO: fine tune model
    # TODO: choose subset training set
