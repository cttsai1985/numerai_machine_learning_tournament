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
from sklearn.model_selection import StratifiedKFold, LeavePGroupsOut, GroupKFold, PredefinedSplit
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from lightgbm import LGBMRegressor, LGBMClassifier

from .PerformanceTracker import PerformanceTracker
from . import Helper
from .CustomSplit import TimeSeriesSplitGroups
from .DataManager import DataManager
from .Metrics import spearman_corr
from .LGBMUtils import lgbm_spearman_eval_func, lgbm_mae_eval_func
from .Utils import get_file_hash


# Scikit-Learn
def spearman_eval_scorer(y: np.ndarray, y_pred: np.ndarray, **kwargs):
    return spearman_corr(y, y_pred)


sklearn_spearman_scorer = make_scorer(
    spearman_eval_scorer, greater_is_better=True, needs_proba=False, needs_threshold=False, )

_available_cv_splitter: Dict[str, Callable] = dict([
    ("StratifiedKFold", StratifiedKFold),
    ("LeavePGroupsOut", LeavePGroupsOut),
    ("GroupKFold", GroupKFold),
    ("PredefinedSplit", PredefinedSplit),
    ("TimeSeriesSplitGroups", TimeSeriesSplitGroups),
])

_available_model_gen: Dict[str, Callable] = dict([
    ("LGBMRegressor", LGBMRegressor),
    ("LGBMClassifier", LGBMClassifier),
])

_available_objective_func: Dict[str, Callable] = dict([
])

_available_evaluation_func: Dict[str, Callable] = dict([
    ("lightgbm_neg_spearman_corr", lgbm_spearman_eval_func),
    ("lightgbm_mean_absolute_error", lgbm_mae_eval_func)
])

_available_sklearn_scorer: Dict[str, Callable] = dict([
    ("sklearn_spearman_scorer", sklearn_spearman_scorer),
])


class BaseSolutionConfigs:
    def __init__(
            self, root_resource_path: str, configs_file_path: str = "configs.yaml", eval_data_type: str = None, ):
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
        self.num_class: int = 1

        self.model_name: str = "foobar"

        self.label2index: Dict[int, float] = dict()
        self.index2label: Dict[float, int] = dict()

        self._load_yaml_configs(configs_file_path)
        self._load_feature_columns(feature_columns_file_path=os.path.join(self.output_dir_, "features.json"))

        # initialize
        self.input_data_dir: str = os.path.join(self.root_resource_path, self.dataset_name)

        self.scoring_func: Callable = PerformanceTracker().score
        # self._save_yaml_configs()

    def _load_feature_columns(self, feature_columns_file_path: Optional[str] = None):
        default_file_path: str = os.path.join(self.meta_data_dir, "features_numerai.json")
        if feature_columns_file_path is None:
            file_path = default_file_path
            logging.info(f"load feature columns from default location: {feature_columns_file_path}")

        if not all([os.path.exists(feature_columns_file_path), os.path.isfile(feature_columns_file_path)]):
            file_path = default_file_path
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
            logging.info(f"from configs yaml to set attribute {k}: {v}")

        return self

    def _save_yaml_configs(self, output_data_dir: Optional[str] = None):
        raise NotImplementedError

    def load_optimized_params_from_tuner(self, tuner):
        raise NotImplementedError

    @property
    def result_folder_name_(self) -> str:
        if self.configs_hash_str is None:
            self.configs_hash_str = get_file_hash(self.configs_file_path)
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
        raise NotImplementedError

    @property
    def template_cv_splitter_(self) -> BaseCrossValidator:
        raise NotImplementedError

    @property
    def cv_splitter_(self) -> BaseCrossValidator:
        raise NotImplementedError

    @classmethod
    def from_config_file(cls, file_path: str):
        pass


class SolutionConfigs(BaseSolutionConfigs):
    def __init__(
            self, root_resource_path: str, configs_file_path: str = "configs.yaml", eval_data_type: str = None,
            num_boost_round: int = 2000):
        super().__init__(
            root_resource_path=root_resource_path, configs_file_path=configs_file_path, eval_data_type=eval_data_type)

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
        self._configure_solution_from_yaml()
        # self._save_yaml_configs()

    def _configure_solution_from_yaml(self, ):
        self.model_gen = _available_model_gen.get(self.model_gen_query)
        self.template_cv_splitter_gen = _available_cv_splitter.get(self.template_cv_splitter_gen_query)
        self.fobj = _available_objective_func.get(self.fobj_gen)
        self.feval = _available_evaluation_func.get(self.feval_gen)
        self.scorer_func = _available_sklearn_scorer.get(self.scorer_func_query)

        if "Classifier" in self.model_gen_query:
            logging.info(f"enable label2index mapping: {self.label2index}, index2label mapping: {self.index2label}")
            self.label2index: Dict[int, float] = {0: .0, 1: .25, 2: .5, 3: .75, 4: 1.}
            self.index2label: Dict[float, int] = {v: k for k, v in self.label2index.items()}

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
            "param_distributions": self.param_distributions,
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

    def load_optimized_params_from_tuner(self, tuner):
        self.model_best_params = tuner.load_optimized_params().params_
        logging.info(f"load best params through tuner: {self.model_best_params}")
        return self

    @property
    def unfitted_model_(self) -> BaseEstimator:
        if self.model_best_params is None:
            logging.info(f"generate model {self.model_gen} with base parameters: {self.base_params}")
            return self.model_gen(**self.base_params)

        logging.info(f"generate model {self.model_gen} with best fitted parameters: {self.model_best_params}")
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


class EnsembleSolutionConfigs(BaseSolutionConfigs):
    def __init__(
            self, root_resource_path: str, configs_file_path: str = "configs.yaml", eval_data_type: str = None, ):
        super().__init__(
            root_resource_path=root_resource_path, configs_file_path=configs_file_path, eval_data_type=eval_data_type)

        self.model_name: str = "ensemble_base"
        self.ensemble_model_configs: Optional[List[SolutionConfigs]] = None
        self.model_dirs: Optional[List[str]] = None

        self._load_yaml_configs(configs_file_path)
        self._configure_solution_from_yaml()
        # self._save_yaml_configs()

    def _configure_solution_from_yaml(self, ):
        _configs = list()
        for config in self.ensemble_model_configs:
            if not (os.path.exists(config) and os.path.isfile(config)):
                logging.error(f"configs does not exist: {config}")
                raise FileExistsError(f"configs does not exist: {config}")

            _config = SolutionConfigs(
                root_resource_path=self.root_resource_path, configs_file_path=config,
                eval_data_type=self.eval_data_type, )
            _configs.append(_config)
            logging.info(f"load configs: {config}, from location: {_config.output_dir_}")

        self.ensemble_model_configs = _configs
        self.model_dirs = [config.output_dir_ for config in _configs]
        logging.info(f"load {len(self.ensemble_model_configs)} configs")
        return self

    def _save_yaml_configs(self, output_data_dir: Optional[str] = None):
        raise NotImplementedError

    @property
    def unfitted_model_(self) -> BaseEstimator:
        raise NotImplementedError

    @property
    def template_cv_splitter_(self) -> BaseCrossValidator:
        raise NotImplementedError

    @property
    def cv_splitter_(self) -> BaseCrossValidator:
        raise NotImplementedError

    @classmethod
    def from_config_file(cls, file_path: str):
        pass
