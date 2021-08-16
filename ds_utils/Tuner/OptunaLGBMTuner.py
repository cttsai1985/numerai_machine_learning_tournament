import sys
import os
import json
import argparse
import logging
import lightgbm as lgb
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from optuna.integration import OptunaSearchCV

from .OptunaLGBMIntegration import OptunaLightGBMTunerCV
from ..DefaultConfigs import RefreshLevel
from ..SolutionConfigs import SolutionConfigs


class _BaseOptunaTuner:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", cv_splitter: BaseCrossValidator,
            base_params: Dict[str, Any], seed: int = 42, working_dir: Optional[str] = None):

        self.refresh_level: RefreshLevel = refresh_level
        self.refresh_level_criterion: RefreshLevel = RefreshLevel("hyper_parameters")
        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        self.is_tuned: bool = False

        self.data_manager: "DataManager" = data_manager
        self.base_params: Dict[str, Any] = base_params

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter

        self.seed: int = seed
        self.best_params: Optional[Dict[str, Any]] = None

    def _run(self, data_type: str = "training"):
        raise NotImplementedError()

    def run(self, data_type: Optional[str] = None):
        if not data_type:
            data_type = "training"

        if self.is_tuned and self.refresh_level > self.refresh_level_criterion:
            logging.info(f"skip due to refresh level ({self.refresh_level}) higher than {self.refresh_level_criterion}")
            return self

        self._run(data_type=data_type)
        self.save_optimized_params()
        return self

    @property
    def params_filepath_(self) -> str:
        return os.path.join(self.working_dir, "model_optimized_params.json")

    @property
    def params_(self) -> Dict[str, Any]:
        if not self.is_tuned:
            logging.warning(f"hyper-parameters is not tuned")
            return self.base_params

        return self.best_params

    def save_optimized_params(self):
        with open(self.params_filepath_, "w") as fp:
            json.dump(self.best_params, fp)
            logging.info(f"save best hyper-parameters: {self.best_params} to {self.params_filepath_}")

        return self

    def load_optimized_params(self):
        if not (os.path.exists(self.params_filepath_) and os.path.isfile(self.params_filepath_)):
            logging.info(f"optimized hyper-parameters not exist: {self.params_filepath_}")
            return self

        with open(self.params_filepath_, "r") as fp:
            self.best_params = json.load(fp)
            logging.info(f"load best hyper-parameters: {self.best_params} from {self.params_filepath_}")

        self.is_tuned = True
        return self

    @classmethod
    def from_configs(cls, args: argparse.Namespace, configs: SolutionConfigs, output_data_path: str):
        return cls(
            refresh_level=args.refresh_level, data_manager=configs.data_manager_, cv_splitter=configs.cv_splitter_,
            base_params=configs.model_base_params, seed=configs.model_tuning_seed, working_dir=output_data_path)


class OptunaSklearnTuner(_BaseOptunaTuner):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", cv_splitter: BaseCrossValidator,
            base_params: Dict[str, Any], param_distributions: Dict[str, Any], model_gen: BaseEstimator, seed: int = 42,
            scorer_func: Optional[Callable[..., Any]] = None, n_trials: int = 10, working_dir: Optional[str] = None):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, cv_splitter=cv_splitter, base_params=base_params,
            seed=seed, working_dir=working_dir)

        self.model_gen: BaseEstimator = model_gen
        self.scorer_func: Optional[Callable[..., Any]] = scorer_func
        self.param_distributions = param_distributions
        self.n_trials: int = n_trials

    def _process_param_distributions(self, param_distributions):
        raise NotImplementedError()

    def _run(self, data_type: str = "training"):
        train_x, train_y, train_group = self.data_manager.get_data_helper_by_type(data_type).data_
        if self.data_manager.has_cast_mapping(cast_type="label2index"):
            self.base_params["num_class"] = train_y.nunique()

        tuner = OptunaSearchCV(
            self.model_gen(**self.base_params), self.param_distributions, cv=self.cv_splitter, enable_pruning=False,
            n_jobs=1, n_trials=10, random_state=self.seed, refit=False, return_train_score=True,
            scoring=self.scorer_func, study=None, subsample=1.0, timeout=None, verbose=0)

        train_y = self.data_manager.cast_target(train_y, cast_type="label2index")
        tuner.fit(train_x, train_y)

        self.best_params = tuner.best_params
        self.best_params["n_estimators"] = self.num_boost_round
        logging.info(f"Best score from Tuning: {tuner.best_score:.6f} with hyper-parameters: {self.best_params}")
        self.is_tuned = True
        return self

    @classmethod
    def from_configs(cls, args: argparse.Namespace, configs: SolutionConfigs, output_data_path: str):
        return cls(
            refresh_level=args.refresh_level, data_manager=configs.data_manager_, cv_splitter=configs.cv_splitter_,
            model_gen=configs.model_gen, base_params=configs.model_base_params,
            param_distributions=configs.param_distributions, scorer_func=configs.scorer_func, n_trials=configs.n_trials,
            seed=configs.model_tuning_seed, working_dir=output_data_path)


class OptunaLGBMTuner(_BaseOptunaTuner):
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: "DataManager", cv_splitter: BaseCrossValidator,
            base_params: Dict[str, Any], param_distributions: Dict[str, Dict[str, Any]], num_boost_round: int = 2000,
            early_stopping_rounds: int = 100, seed: int = 42, fobj: Optional[Callable[..., Any]] = None,
            feval: Optional[Callable[..., Any]] = None, working_dir: Optional[str] = None):
        super().__init__(
            refresh_level=refresh_level, data_manager=data_manager, cv_splitter=cv_splitter, base_params=base_params,
            seed=seed, working_dir=working_dir)

        self.fobj: Optional[Callable[..., Any]] = fobj
        self.feval: Optional[Callable[..., Any]] = feval

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds

        self.param_distributions: Dict[str, Dict[str, Any]] = param_distributions

    def _run(self, data_type: str = "training"):
        train_x, train_y, train_group = self.data_manager.get_data_helper_by_type(data_type).data_

        if self.data_manager.has_cast_mapping(cast_type="label2index"):
            self.base_params["num_class"] = train_y.nunique()
        train_y = self.data_manager.cast_target(train_y, cast_type="label2index")
        tuner = OptunaLightGBMTunerCV(
            self.base_params, param_distributions=self.param_distributions,
            train_set=lgb.Dataset(train_x, label=train_y), fobj=self.fobj, feval=self.feval,
            num_boost_round=self.num_boost_round, early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100, folds=self.cv_splitter, seed=self.seed, time_budget=None, study=None,
            model_dir=self.working_dir, )

        tuner.run()
        self.best_params = tuner.best_params
        self.best_params["n_estimators"] = self.num_boost_round
        logging.info(f"Best score from Tuning: {tuner.best_score:.6f} with hyper-parameters: {self.best_params}")
        self.is_tuned = True
        return self

    @classmethod
    def from_configs(cls, args: argparse.Namespace, configs: SolutionConfigs, output_data_path: str):
        return cls(
            refresh_level=args.refresh_level, data_manager=configs.data_manager_, cv_splitter=configs.cv_splitter_,
            base_params=configs.model_base_params, param_distributions=configs.param_distributions,
            num_boost_round=configs.num_boost_round, early_stopping_rounds=configs.early_stopping_rounds,
            seed=configs.model_tuning_seed, fobj=configs.fobj, feval=configs.feval, working_dir=output_data_path)
