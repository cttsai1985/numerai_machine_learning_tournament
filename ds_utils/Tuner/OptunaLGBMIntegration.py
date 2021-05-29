import copy
import json
import logging
import numpy as np
import tqdm
import optuna
from typing import Optional, Callable, Any, Dict, List, Tuple, Union, Generator, Iterator
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.integration._lightgbm_tuner.alias import _handling_alias_parameters
from optuna.integration._lightgbm_tuner.optimize import _OptunaObjective, _OptunaObjectiveCV
from optuna.integration.lightgbm import LightGBMTunerCV

# Define key names of `Trial.system_attrs`.
_LGBM_PARAMS_KEY = "lightgbm_tuner:lgbm_params"

# EPS is used to ensure that a sampled parameter value is in pre-defined value range.
_EPS = 1e-12
# Default value of tree_depth, used for upper bound of num_leaves.
_DEFAULT_TUNER_TREE_DEPTH: int = 16
_MIN_TUNER_TREE_DEPTH: int = 4

# Default parameter values described in the official webpage.
_DEFAULT_LIGHTGBM_PARAMETERS = {
    "lambda_l1": 1e-3,
    "lambda_l2": 1e-6,
    "num_leaves": 31,
    "feature_fraction": .05,
    "bagging_fraction": .9,
    "bagging_freq": 1,
    "min_child_samples": 20,
}


def min_num_leave_by_depth(tree_depth: int):
    return 2 * tree_depth - 1


def max_num_leave_by_depth(tree_depth: int):
    return 2 ** tree_depth - 1


def adaptive_func(tree_depth: int):
    return int(max(0, (tree_depth - 1) / 2, ) ** 2)


class _CustomOptunaObjectiveCV(_OptunaObjectiveCV):
    def __init__(
            self, target_param_names: List[str], param_range: Dict[str, Dict[str, float]],
            lgbm_params: Dict[str, Any], train_set: "lgb.Dataset", lgbm_kwargs: Dict[str, Any], best_score: float,
            step_name: str, model_dir: Optional[str], pbar: Optional[tqdm.tqdm] = None, ):
        super().__init__(
            target_param_names, lgbm_params, train_set, lgbm_kwargs, best_score, step_name, model_dir, pbar=pbar, )
        self._param_range: Dict[str, Dict[str, float]] = param_range

    def _preprocess(self, trial: optuna.trial.Trial) -> None:
        if self.pbar is not None:
            self.pbar.set_description(self.pbar_fmt.format(self.step_name, self.best_score))

        if "lambda_l1" in self.target_param_names:
            self.lgbm_params["lambda_l1"] = np.round(trial.suggest_float("lambda_l1", 1e-3, 100.0, log=True), 3)

        if "lambda_l2" in self.target_param_names:
            self.lgbm_params["lambda_l2"] = np.round(trial.suggest_float("lambda_l2", 1e-6, 10.0, log=True), 6)

        if "num_leaves" in self.target_param_names:
            tree_depth = max(
                _MIN_TUNER_TREE_DEPTH, self.lgbm_params.get("max_depth", _DEFAULT_TUNER_TREE_DEPTH))
            # min tree_depth >= 4
            margin = int(np.log(tree_depth ** 2))
            min_num_leaves = min_num_leave_by_depth(min(_MIN_TUNER_TREE_DEPTH, tree_depth)) + margin

            max_num_leaves = min(
                max_num_leave_by_depth(min(tree_depth, 7)),
                sum(map(max_num_leave_by_depth, range(tree_depth - 3, tree_depth))))
            logging.info(f"num_leaves for max_depth {tree_depth}: in ({min_num_leaves}, {max_num_leaves})")
            self.lgbm_params["num_leaves"] = int(trial.suggest_loguniform("num_leaves", min_num_leaves, max_num_leaves))

        if "feature_fraction" in self.target_param_names:
            # `GridSampler` is used for sampling feature_fraction value.
            param_name = "feature_fraction"
            param = self._param_range[param_name]
            self.lgbm_params[param_name] = np.round(min(
                trial.suggest_float(param_name, param["low"], param["high"] + _EPS), param["high"]), 3)

        if "bagging_fraction" in self.target_param_names:
            # `TPESampler` is used for sampling bagging_fraction value.
            param_name = "bagging_fraction"
            param = self._param_range[param_name]
            self.lgbm_params[param_name] = min(
                trial.suggest_float(param_name, param["low"], param["high"] + _EPS), param["high"])

        if "bagging_freq" in self.target_param_names:
            self.lgbm_params["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

        if "min_child_samples" in self.target_param_names:
            # `GridSampler` is used for sampling min_child_samples value.
            self.lgbm_params["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)


class OptunaLightGBMTunerCV(LightGBMTunerCV):
    def __init__(
            self,
            params: Dict[str, Any],
            train_set: "lgb.Dataset",
            num_boost_round: int = 1000,
            folds: Optional[Union[
                Generator[Tuple[int, int], None, None], Iterator[Tuple[int, int]], "BaseCrossValidator",]] = None,
            nfold: int = 5,
            stratified: bool = True,
            shuffle: bool = True,
            fobj: Optional[Callable[..., Any]] = None,
            feval: Optional[Callable[..., Any]] = None,
            feature_name: str = "auto",
            categorical_feature: str = "auto",
            early_stopping_rounds: Optional[int] = None,
            fpreproc: Optional[Callable[..., Any]] = None,
            verbose_eval: Optional[Union[bool, int]] = True,
            show_stdv: bool = True,
            seed: int = 0,
            callbacks: Optional[List[Callable[..., Any]]] = None,
            time_budget: Optional[int] = None,
            sample_size: Optional[int] = None,
            study: Optional[optuna.study.Study] = None,
            optuna_callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
            verbosity: Optional[int] = None,
            show_progress_bar: bool = True,
            model_dir: Optional[str] = None,
            return_cvbooster: Optional[bool] = None, ):
        super().__init__(
            params, train_set, num_boost_round, folds=folds, nfold=nfold, stratified=stratified, shuffle=shuffle,
            fobj=fobj, feval=feval, feature_name=feature_name, categorical_feature=categorical_feature,
            early_stopping_rounds=early_stopping_rounds, fpreproc=fpreproc, verbose_eval=verbose_eval,
            show_stdv=show_stdv, seed=seed, callbacks=callbacks, time_budget=time_budget, sample_size=sample_size,
            study=study, optuna_callbacks=optuna_callbacks, verbosity=verbosity, show_progress_bar=show_progress_bar,
            model_dir=model_dir, return_cvbooster=return_cvbooster)
        self._param_range: Dict[str, Dict[str, float]] = {
            "feature_fraction": {"low": .05, "high": .2, "step": .05},
            "bagging_fraction": {"low": .6, "high": .95, "step": .01},
            "min_child_samples": {"discrete": [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]}
        }

    def tune_feature_fraction(self, n_trials: int = 7) -> None:
        param_name = "feature_fraction"
        param = self._param_range[param_name]
        param_values = np.linspace(
            param["low"], param["high"], int((param["high"] - param["low"]) / param["step"]) + 1).tolist()
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction")

    def tune_feature_fraction_stage2(self, n_trials: int = 21) -> None:
        param_name: str = "feature_fraction"
        best_feature_fraction = self.best_params[param_name]
        param = self._param_range[param_name]
        param_values = np.round(
            np.linspace(best_feature_fraction - .03, best_feature_fraction + .03, n_trials), 3).tolist()
        param_values = list(filter(lambda val: param["high"] >= val >= param["low"], param_values))
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction_stage2")

    def tune_num_leaves(self, n_trials: int = 25) -> None:
        self._tune_params(["num_leaves"], n_trials, optuna.samplers.TPESampler(), "num_leaves")

    def tune_bagging(self, n_trials: int = 25) -> None:
        self._tune_params(["bagging_fraction", "bagging_freq"], n_trials, optuna.samplers.TPESampler(), "bagging")

    def tune_regularization_factors(self, n_trials: int = 25) -> None:
        self._tune_params(
            ["lambda_l1", "lambda_l2"], n_trials, optuna.samplers.TPESampler(), "regularization_factors", )

    def tune_min_data_in_leaf(self) -> None:
        param_name = "min_child_samples"
        param = self._param_range[param_name]
        param_values = param["discrete"]
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, "min_data_in_leaf")

    def _create_objective(
            self, target_param_names: List[str], train_set: "lgb.Dataset", step_name: str,
            pbar: Optional[tqdm.tqdm], ) -> _OptunaObjective:
        return _CustomOptunaObjectiveCV(
            target_param_names, self._param_range, self.lgbm_params, train_set, self.lgbm_kwargs, self.best_score,
            step_name=step_name, model_dir=self._model_dir, pbar=pbar, )

    def run(self) -> None:
        """Perform the hyperparameter-tuning with given parameters."""
        verbosity = self.auto_options["verbosity"]
        if verbosity is not None:
            if verbosity > 1:
                optuna.logging.set_verbosity(optuna.logging.DEBUG)
            elif verbosity == 1:
                optuna.logging.set_verbosity(optuna.logging.INFO)
            elif verbosity == 0:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
            else:
                optuna.logging.set_verbosity(optuna.logging.CRITICAL)

        # Handling aliases.
        _handling_alias_parameters(self.lgbm_params)

        # Sampling.
        self.sample_train_set()

        self.tune_feature_fraction()
        logging.info(f"current best after tune_feature_fraction: {self.best_params}")
        self.tune_num_leaves()
        logging.info(f"current best after tune_num_leaves: {self.best_params}")
        self.tune_bagging()
        logging.info(f"current best after tune_bagging: {self.best_params}")
        self.tune_feature_fraction_stage2()
        logging.info(f"current best after tune_feature_fraction_stage2: {self.best_params}")
        self.tune_regularization_factors()
        logging.info(f"current best after tune_regularization_factors: {self.best_params}")
        self.tune_min_data_in_leaf()
        logging.info(f"current best after tune_min_data_in_leaf: {self.best_params}")

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return parameters of the best booster."""
        try:
            return json.loads(self.study.best_trial.system_attrs[_LGBM_PARAMS_KEY])
        except ValueError:
            # Return the default score because no trials have completed.
            params = copy.deepcopy(_DEFAULT_LIGHTGBM_PARAMETERS)
            # self.lgbm_params may contain parameters given by users.
            params.update(self.lgbm_params)
            return params
