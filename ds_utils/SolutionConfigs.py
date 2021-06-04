import os
import yaml
import json
import logging
import hashlib
import lightgbm
import numpy as np
import pandas as pd
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
from functools import partial
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor

from .PerformanceTracker import PerformanceTracker
from . import Helper
from .DataManager import DataManager


def spearman_corr(y: np.ndarray, y_preds: np.ndarray, **kwargs):
    return spearmanr(y, pd.Series(y_preds).rank(pct=True, method="first"))[0]


def lgbm_spearman_eval_func(preds: np.ndarray, train_data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    return "neg_spearman_corr", -1. * spearman_corr(train_data.get_label(), preds), False


def spearman_eval_scorer(y: np.ndarray, y_pred: np.ndarray, **kwargs):
    return spearmanr(y, pd.Series(y_pred).rank(pct=True, method="first"))[0]


sklearn_spearman_scorer = make_scorer(
    spearman_eval_scorer, greater_is_better=True, needs_proba=False, needs_threshold=False,)


_available_cv_splitter: Dict[str, Callable] = dict([
    ("StratifiedKFold", StratifiedKFold),
    ("PredefinedSplit", PredefinedSplit)])

_available_model_gen: Dict[str, Callable] = dict([
    ("LGBMRegressor", LGBMRegressor),
])

_available_objective_func: Dict[str, Callable] = dict([
])

_available_evaluation_func: Dict[str, Callable] = dict([
    ("lightgbm_neg_spearman_corr", lgbm_spearman_eval_func)
])


_available_sklearn_scorer: Dict[str, Callable] = dict([
    ("sklearn_spearman_scorer", sklearn_spearman_scorer),
])


class SolutionConfigs:
    def __init__(
            self, root_resource_path: str, configs_file_path: str = "configs.yaml", eval_data_type: str = None,
            num_boost_round: int = 2000):
        self.configs_file_path: Optional[str] = configs_file_path
        self.configs_hash_str: Optional[str] = None

        self.root_resource_path: str = root_resource_path
        self.meta_data_dir: str = os.path.join(self.root_resource_path, "metadata")

        self.eval_data_type: str = eval_data_type if eval_data_type is not None else "training"

        self.data_manager: Optional["DataManager"] = None

        self.dataset_name: Optional[str] = "latest_tournament_datasets"
        self.data_mapping: Optional[Dict[str, str]] = None
        self.column_target: Optional[str] = "target"
        self.columns_group: Optional[List[str]] = ["era"]
        self.columns_feature: Optional[List[str]] = None

        self.template_cv_splitter_gen_query: str = "StratifiedKFold"
        self.template_cv_splitter_gen: BaseCrossValidator = StratifiedKFold
        self.template_cv_splitter_params: Dict[str, Any] = {"n_splits": 5, "shuffle": True, "random_state": 42}
        self.cv_splitter: Optional[BaseCrossValidator] = None

        self.model_name: str = "baseline_lightgbm_optuna_fair_mae"
        self.model_gen_query: str = "LGBMRegressor"
        self.model_gen: BaseEstimator = LGBMRegressor
        self.model_base_params: Dict[str, Any] = {
            "objective": "fair",
            "metric": ["rmse", "neg_spearman_corr"],
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": .01,
            "max_depth": 24,
            "n_jobs": 12
        }
        self.model_best_params: Optional[Dict[str, Any]] = None
        self.early_stopping_rounds: Optional[int] = None
        self.num_boost_round: int = num_boost_round

        self.scorer_func_query: str = "sklearn_spearman_scorer"
        self.scorer_func: Optional[Callable[..., Any]] = None
        self.param_distributions: Optional[Dict[str, Any]] = None
        self.n_trials: int = 10
        self.model_tuning_seed: int = 42
        self.fobj_gen: Optional[str] = None
        self.fobj: Optional[Callable[..., Any]] = None
        self.feval_gen: Optional[str] = None
        self.feval: Optional[Callable[..., Any]] = None

        self._load_yaml_configs(configs_file_path)

        # initialize
        self.input_data_dir: str = os.path.join(self.root_resource_path, self.dataset_name)

        self.scoring_func: Callable = PerformanceTracker().score
        # self._save_yaml_configs()

    def _load_feature_columns(self, feature_columns_file_path: Optional[str] = None):
        if feature_columns_file_path is None:
            file_path = os.path.join(self.meta_data_dir, "features_numerai.json")
            logging.info(f"load feature columns from default location: {feature_columns_file_path}")

        if not all([os.path.exists(feature_columns_file_path), os.path.isfile(feature_columns_file_path)]):
            file_path = os.path.join(self.meta_data_dir, "features_numerai.json")
            logging.info(f"load feature columns from default location: {file_path}")

        with open(file_path, "r") as fp:
            self.columns_feature = json.load(fp)

        logging.info(f"load using {len(self.columns_feature)} features from {file_path}")
        return self

    def _load_yaml_configs(self, configs_file_path: str):
        if not all([os.path.exists(configs_file_path), os.path.isfile(configs_file_path)]):
            logging.info(f"configs does not exist: {configs_file_path}")
            return self

        with open(configs_file_path, 'r') as fp:
            dict_file = yaml.load(fp, Loader=yaml.FullLoader)

        for k, v in dict_file.items():
            setattr(self, k, v)
            logging.info(f"set attribute {k}: {v}")

        self._load_feature_columns(feature_columns_file_path=os.path.join(self.output_dir_, "features.json"))

        self.model_gen = _available_model_gen.get(self.model_gen_query)
        self.template_cv_splitter_gen = _available_cv_splitter.get(self.template_cv_splitter_gen_query)
        self.fobj = _available_objective_func.get(self.fobj_gen)
        self.feval = _available_evaluation_func.get(self.feval_gen)
        self.scorer_func = _available_sklearn_scorer.get(self.scorer_func_query)
        return self

    def _save_yaml_configs(self, output_data_dir: Optional[str] = None):
        dict_file = {
            "dataset_name": self.dataset_name,
            "data_mapping": self.data_mapping,
            "column_target": self.column_target,
            "columns_group": self.columns_group,
            "model_name": self.model_name,
            "model_gen_query": self.model_gen_query,
            "model_base_params": self.model_base_params,
            "model_best_params": self.model_best_params,
            "model_tuning_seed": self.model_tuning_seed,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "template_cv_splitter_gen_query": self.template_cv_splitter_gen_query,
            "template_cv_splitter_params": self.template_cv_splitter_params,
            "feval_gen": self.feval_gen,
            "fobj_gen": self.fobj_gen,
            "scorer_func_query": self.scorer_func_query,
            "n_trials": self.n_trials
        }

        if output_data_dir is None:
            output_data_dir = self.output_dir_

        with open(os.path.join(output_data_dir, "configs.yaml"), 'w') as fp:
            yaml.dump(dict_file, fp)

        return self

    @property
    def result_folder_name_(self) -> str:
        if self.configs_hash_str is None:
            self.configs_hash_str = hashlib.md5(open(self.configs_file_path, 'rb').read()).hexdigest()[:8]

        return "_".join([self.model_name, self.configs_hash_str])

    @property
    def output_dir_(self) -> str:
        return os.path.join(self.root_resource_path, self.result_folder_name_)

    @property
    def data_manager_(self) -> "DataManager":
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
