import sys
import os
import json
import argparse
import logging
import lightgbm as lgb
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
from sklearn.model_selection import BaseCrossValidator

from .OptunaLGBMIntegration import OptunaLightGBMTunerCV
from ..DefaultConfigs import RefreshLevel
from ..DataManager import DataManager
from ..Solution import SolutionConfigs


class OptunaLGBMTuner:
    def __init__(
            self, refresh_level: RefreshLevel, data_manager: DataManager, cv_splitter: BaseCrossValidator,
            base_params: Dict[str, Any], num_boost_round: int = 2000, early_stopping_rounds: int = 100, seed: int = 42,
            fobj: Optional[Callable[..., Any]] = None, feval: Optional[Callable[..., Any]] = None,
            working_dir: Optional[str] = None):
        self.refresh_level: RefreshLevel = refresh_level
        self.refresh_level_criterion: RefreshLevel = RefreshLevel("hyper_parameters")
        self.working_dir: str = working_dir if working_dir else "./"
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        self.is_tuned: bool = False

        self.fobj: Optional[Callable[..., Any]] = fobj
        self.feval: Optional[Callable[..., Any]] = feval

        self.data_manager = data_manager
        self.base_params = base_params
        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds

        self.cv_splitter: Optional[BaseCrossValidator] = cv_splitter

        self.seed: int = seed
        self.best_params: Optional[Dict[str, Any]] = None

    def _run(self, data_type: str = "training"):
        train_x, train_y, train_group = self.data_manager.get_data_helper_by_type(data_type).data_
        tuner = OptunaLightGBMTunerCV(
            self.base_params, train_set=lgb.Dataset(train_x, label=train_y), fobj=self.fobj, feval=self.feval,
            num_boost_round=self.num_boost_round, early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100, folds=self.cv_splitter, seed=self.seed, time_budget=None, study=None,
            model_dir=self.working_dir, )

        tuner.run()
        self.best_params = tuner.best_params
        self.best_params["n_estimators"] = self.num_boost_round
        logging.info(f"Best score from Tuning: {tuner.best_score:.6f} with hyper-parameters: {self.best_params}")
        self.is_tuned = True
        return self

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
        if not (os.path.exists(self.params_filepath_) or os.path.isfile(self.params_filepath_)):
            logging.info(f"optimized hyperparameters not exist: {self.params_filepath_}")
            return self

        with open(self.params_filepath_, "r") as fp:
            self.best_params = json.load(fp)
            logging.info(f"load best hyper-parameters: {self.best_params}")

        self.is_tuned = True
        return self

    @classmethod
    def from_configs(cls, args: argparse.Namespace, configs: SolutionConfigs, output_data_path: str):
        return cls(
            refresh_level=args.refresh_level, data_manager=configs.data_manager_, cv_splitter=configs.cv_splitter_,
            base_params=configs.model_base_params, num_boost_round=configs.num_boost_round, early_stopping_rounds=100,
            seed=configs.model_tuning_seed, fobj=configs.fobj, feval=configs.feval, working_dir=output_data_path)
