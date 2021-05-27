import sys
import os
import json
import argparse
import logging
import hashlib

import lightgbm
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr

from lightgbm import LGBMRegressor

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import PerformanceTracker, Solution, DataManager, OptunaLGBMTuner, Helper
from ds_utils import RefreshLevel


def spearman_eval_func(preds: np.ndarray, train_data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    score = spearmanr(train_data.get_label(), pd.Series(preds).rank(pct=True, method="first"))[0]
    return "neg_spearman_corr", -1. * score, False


class SolutionConfigs:
    def __init__(
            self, root_resource_path: str, eval_data_type: str = None, num_boost_round: int = 2000):
        # TODO: Load data configs and modeling configs
        dataset_name: str = "latest_tournament_datasets"
        self.root_resource_path: str = root_resource_path
        self.input_data_dir: str = os.path.join(self.root_resource_path, dataset_name)

        data_file_types: List[str] = ["training", "validation", "test", "live", "tournament"]
        data_filenames: List[str] = [
            "numerai_training_data.parquet", "numerai_validation_data.parquet", "numerai_test_data.parquet",
            "numerai_live_data.parquet", "numerai_tournament_data.parquet"]

        self.data_mapping: Optional[Dict[str, str]] = {k: v for k, v in zip(data_file_types, data_filenames)}
        self.eval_data_type: str = eval_data_type if eval_data_type is not None else "training"

        self.column_target: str = "target"
        self.columns_group: List[str] = ["era"]

        self.meta_data_path: str = os.path.join(self.root_resource_path, "metadata")
        feature_columns_set: str = os.path.join(self.meta_data_path, "features_numerai.json")
        with open(feature_columns_set, "r") as fp:
            columns_feature = json.load(fp)

        self.columns_feature: List[str] = columns_feature

        self.data_manager: Optional[DataManager] = None

        self.template_cv_splitter_gen: BaseCrossValidator = StratifiedKFold
        self.template_cv_splitter_params: Dict[str, Any] = {"n_splits": 5, "shuffle": True, "random_state": 42}
        self.cv_splitter: Optional[BaseCrossValidator] = None

        self.model_name: str = "lightgbm_optuna_fair_mae"
        self.model_gen: BaseEstimator = LGBMRegressor
        self.model_base_params: Dict[str, Any] = {
            # "objective": "regression",
            "objective": "fair",  # "regression",  # "huber", "fair"
            "metric": ["rmse", "neg_spearman_corr"],
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": .01,
            "max_depth": 24,
            "n_jobs": 12
        }
        self.model_best_params: Optional[Dict[str, Any]] = None
        self.model_tuning_seed: int = 42
        self.num_boost_round: int = num_boost_round
        self.fobj: Optional[Callable[..., Any]] = None
        self.feval: Optional[Callable[..., Any]] = spearman_eval_func

        self.scoring_func: Callable = PerformanceTracker().score

    @property
    def model_name_(self) -> str:
        return "_".join([
            self.model_name, hashlib.md5(self.input_data_dir.encode()).hexdigest()[:8]])

    @property
    def data_manager_(self) -> DataManager:
        if self.data_manager is None:
            logging.info(f"initialize data manager on {self.input_data_dir}")
            self.data_manager = DataManager.from_configs(self)

        return self.data_manager

    @property
    def unfitted_model_(self) -> BaseEstimator:
        if self.model_best_params is None:
            logging.info(f"generate model {self.model_gen} with base parameters: {self.base_params}")
            return self.model_gen(**self.base_params)

        logging.info(f"generate model {self.model_gen} with base parameters: {self.model_best_params}")
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
    parser.add_argument(
        "--num-boost-round", type=int, default=2000, help="number of boost round for searching hyper-parameters")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    if args.debug:
        args.eval_data_type = "validation"
        args.num_boost_round = 5

    args.refresh_level = RefreshLevel(args.refresh_level)
    if args.compute_all:
        args.compute_hpo = True
        args.compute_eval = True
        args.compute_infer = True
    return args


def main(args: argparse.Namespace):
    root_resource_path: str = "../input/numerai_tournament_resource/"
    modeling_type: str = "baseline"

    configs = SolutionConfigs(
        root_resource_path=root_resource_path, eval_data_type=args.eval_data_type,
        num_boost_round=args.num_boost_round)

    output_data_path: str = os.path.join(root_resource_path, modeling_type, configs.model_name_)
    Path(output_data_path).mkdir(parents=True, exist_ok=True)

    tuner = OptunaLGBMTuner.from_configs(
        args=args, configs=configs, output_data_path=output_data_path).load_optimized_params()
    if args.compute_hpo:
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
    # TODO: building
    # TODO: model helper
    # TODO: parameterize script, half way
    # TODO: Post processing
    # TODO: choose subset training set
    # TODO: fine tune model
    # TODO: add select era for model analytics

    # TODO: experimenting
    # TODO: multi class approach
