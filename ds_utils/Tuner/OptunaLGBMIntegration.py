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

_SUPPORTED_PARAM_NAMES = [
    "learning_rate",
    "lambda_l1",
    "lambda_l2",
    "num_leaves",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "min_child_samples",
    # "drop_rate",  # DART specific parameters
    # "skip_drop",  # DART specific parameters
    # "max_drop",  # DART specific parameters
    # "top_rate",  # GOSS specific parameters
    # "other_rate",  # GOSS specific parameters
    # "fair_c",  # fair loss specific parameters, default = 1, and > 0
    # "alpha",  # huber and quantile regression loss specific parameters, default = 0.9 and > 0.
]

_ADDITIONAL_SUPPORTED_PARAM_NAMES: Dict[str, List[str]] = {
    ("fair",): ["fair_c"],
    ("huber", "quantile"): ["alpha"],
    ("goss",): ["top_rate", "other_rate"],
    ("dart",): ["drop_rate", "skip_drop", "max_drop"],
}

# TO DROP PARAMETERS
_PARAMS_TO_CHECK: List[Tuple[str, List[str]]] = [
    ("goss", ["bagging_fraction", "bagging_freq"]),
    ("rf", ["learning_rate"])
]


def _modify_supported_parameters(param_values: List[Any]) -> List[str]:
    for k, v in _ADDITIONAL_SUPPORTED_PARAM_NAMES.items():
        for pv in param_values:
            if pv not in k:
                continue

            _ADDITIONAL_PARAMS = _ADDITIONAL_SUPPORTED_PARAM_NAMES.get(k, list())
            if not _ADDITIONAL_PARAMS:
                continue

            _SUPPORTED_PARAM_NAMES.extend(_ADDITIONAL_PARAMS)
            logging.info(f"Adding Light GBM parameters: {', '.join(_ADDITIONAL_PARAMS)}")

    logging.info(f"Allowed to tuning Light GBM parameters: {_SUPPORTED_PARAM_NAMES}")
    return _SUPPORTED_PARAM_NAMES


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

    def _check_target_names_supported(self) -> None:
        not_supported_parameters = list(filter(lambda x: x not in _SUPPORTED_PARAM_NAMES, self.target_param_names))
        if not_supported_parameters:
            raise NotImplementedError(f"Parameter `{', '.join(not_supported_parameters)}` is not supported for tuning.")

    def _preprocess(self, trial: optuna.trial.Trial) -> None:
        if self.pbar is not None:
            self.pbar.set_description(self.pbar_fmt.format(self.step_name, self.best_score))

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

        self._suggest_float("lambda_l1", trial, round_digits=3)
        self._suggest_float("lambda_l2", trial, round_digits=6)
        # `GridSampler` is used for sampling feature_fraction value.
        self._suggest_float("feature_fraction", trial, round_digits=3)
        # `TPESampler` is used for sampling bagging_fraction value.
        self._suggest_float("bagging_fraction", trial, round_digits=None)
        # `GridSampler` is used for sampling bagging_freq value.
        self._suggest_categorical("bagging_freq", trial)
        # `GridSampler` is used for sampling min_child_samples value.
        self._suggest_categorical("min_child_samples", trial)
        # `GridSampler` is used for sampling learning_rate value.
        self._suggest_categorical("learning_rate", trial)
        # `TPESampler` is used for sampling dart specific parameters.
        self._suggest_float("drop_rate", trial, round_digits=3)
        self._suggest_float("skip_drop", trial, round_digits=3)
        self._suggest_categorical("max_drop", trial)
        # `GridSampler` is used for GOSS loss tuning.
        self._suggest_categorical("top_rate", trial)
        self._suggest_categorical("other_rate", trial)
        # `GridSampler` is used for fair loss tuning.
        self._suggest_categorical("fair_c", trial)
        # `GridSampler` is used for huber loss tuning.
        self._suggest_categorical("alpha", trial)

    def _suggest_float(
            self, param_name: str, trial: optuna.trial.Trial, round_digits: Optional[float] = None) -> None:
        if param_name not in self.target_param_names:
            return

        param = self._param_range[param_name]
        suggestion = min(
            trial.suggest_float(param_name, param["low"], param["high"] + _EPS, log=param.get("log", False)),
            param["high"])
        if round_digits:
            suggestion = np.round(suggestion, round_digits)

        self.lgbm_params[param_name] = suggestion
        return

    def _suggest_categorical(self, param_name: str, trial: optuna.trial.Trial) -> None:
        if param_name not in self.target_param_names:
            return

        param = self._param_range[param_name]
        self.lgbm_params[param_name] = trial.suggest_categorical(param_name, param["discrete"])
        return


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

        self._param_range: Dict[str, Dict[str, Any]] = {
            # loss function tuning
            "fair_c": {"discrete": [0.5, 0.75, 1, 1.25, 1.5, 2]},
            "alpha": {"discrete": [0.5, 0.8, 0.9, 0.95, 1., 2.]},

            "learning_rate": {"discrete": [.002, .005, .01, .02, .05, .1]},

            "feature_fraction": {"low": .05, "high": .2, "step": .05},
            "bagging_fraction": {"low": .6, "high": .95, "step": .01},
            "bagging_freq": {"discrete": [1, 2, 3, 4, 5, 6, 7]},

            # regularization
            "lambda_l1": {"low": 1e-3, "high": 100., "log": True},
            "lambda_l2": {"low": 1e-6, "high": 10., "log": True},
            "min_child_samples": {"discrete": [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]},

            # goss specific parameters
            "top_rate": {"discrete": [.15, .2, .25, .3, .4]},
            "other_rate": {"discrete": [.05, 0.075, .1, .125, .15, .2]},

            # dart specific parameters
            "max_drop": {"discrete": [0, 0, 10, 25, 40, 50, 60, 75, 90, 100]},
            "drop_rate": {"low": .05, "high": .20, "log": False},
            "skip_drop": {"low": .25, "high": .75, "log": False},
        }

    def _tune_discrete_with_grid_template(self, param_name: str, tuning_task_name: str):
        param = self._param_range[param_name]
        param_values = param["discrete"]
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, tuning_task_name)

    def tune_loss_function(self, n_trials: int = 5) -> bool:
        _param = self.lgbm_params.get("objective")
        if _param not in ["fair", "huber"]:
            return False

        _objective = {"fair": "fair_c", "huber": "alpha", "quantile": "alpha", }
        _param_key = _objective.get(_param, None)
        self._tune_discrete_with_grid_template(param_name=_param_key, tuning_task_name="loss function tuning")
        return True

    def tune_goss_parameters(self, n_trials: int = 5) -> bool:
        if "goss" not in self.lgbm_params.values():
            return False

        self._tune_params(["top_rate", "other_rate"], n_trials, optuna.samplers.TPESampler(), "goss tuning")
        return True

    def tune_dart_parameters(self, n_trials: int = 20) -> bool:
        if "dart" not in self.lgbm_params.values():
            return False

        self._tune_params(["drop_rate", "skip_drop", "max_drop"], n_trials, optuna.samplers.TPESampler(), "dart tuning")
        return True

    def tune_feature_fraction(self, n_trials: int = 7) -> None:
        param_name = "feature_fraction"
        param = self._param_range[param_name]
        param_values = np.linspace(
            param["low"], param["high"], int((param["high"] - param["low"]) / param["step"]) + 1).tolist()
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction")

    def tune_feature_fraction_stage2(self, n_trials: int = 5) -> None:
        param_name: str = "feature_fraction"
        best_feature_fraction = self.best_params[param_name]
        param = self._param_range[param_name]
        param_values = np.round(
            np.linspace(best_feature_fraction - .03, best_feature_fraction + .03, n_trials), 3).tolist()
        param_values = list(filter(lambda val: param["high"] >= val >= param["low"], param_values))
        sampler = optuna.samplers.GridSampler({param_name: param_values})
        self._tune_params([param_name], len(param_values), sampler, "feature_fraction_stage2")

    def tune_num_leaves(self, n_trials: int = 20) -> None:
        self._tune_params(["num_leaves"], n_trials, optuna.samplers.TPESampler(), "num_leaves")

    def tune_regularization_factors(self, n_trials: int = 20) -> None:
        self._tune_params(
            ["lambda_l1", "lambda_l2"], n_trials, optuna.samplers.TPESampler(), "regularization_factors", )

    def tune_min_data_in_leaf(self) -> None:
        self._tune_discrete_with_grid_template(param_name="min_child_samples", tuning_task_name="min_data_in_leaf")

    def tune_bagging(self, n_trials: int = 10) -> bool:  # TODO: find the correct condition for skip
        if "goss" in self.lgbm_params.values():
            return False

        self._tune_params(["bagging_fraction", "bagging_freq"], n_trials, optuna.samplers.TPESampler(), "bagging")
        return True

    def tune_learning_rate(self) -> bool:   # TODO: find the correct condition for skip
        if "rf" in self.lgbm_params.values():
            return False

        self._tune_discrete_with_grid_template(param_name="learning_rate", tuning_task_name="learning_rate")
        return True

    def _create_objective(
            self, target_param_names: List[str], train_set: "lgb.Dataset", step_name: str,
            pbar: Optional[tqdm.tqdm], ) -> _OptunaObjective:
        return _CustomOptunaObjectiveCV(
            target_param_names, self._param_range, self.lgbm_params, train_set, self.lgbm_kwargs, self.best_score,
            step_name=step_name, model_dir=self._model_dir, pbar=None, )

    def run(self) -> None:
        """Perform the hyper-parameters-tuning with given parameters."""
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

        status = self.tune_goss_parameters()
        if status:
            logging.info(f"current best after tune_goss_parameters: {self.best_params}")

        status = self.tune_learning_rate()
        if status:
            logging.info(f"current best after tune_learning_rate: {self.best_params}")

        status = self.tune_loss_function()
        if status:
            logging.info(f"current best after tune_loss_function: {self.best_params}")

        self.tune_feature_fraction()
        logging.info(f"current best after tune_feature_fraction: {self.best_params}")
        self.tune_num_leaves()
        logging.info(f"current best after tune_num_leaves: {self.best_params}")

        status = self.tune_bagging()
        if status:
            logging.info(f"current best after tune_bagging: {self.best_params}")

        self.tune_feature_fraction_stage2()
        logging.info(f"current best after tune_feature_fraction_stage2: {self.best_params}")

        status = self.tune_dart_parameters()
        if status:
            logging.info(f"current best after tune_dart_parameters: {self.best_params}")

        self.tune_regularization_factors()
        logging.info(f"current best after tune_regularization_factors: {self.best_params}")
        self.tune_min_data_in_leaf()
        logging.info(f"current best after tune_min_data_in_leaf: {self.best_params}")

    @staticmethod
    def _drop_ineffective_params(params: Dict[str, Any], param_value: Any, params_to_drop: List[str]) -> Dict[str, Any]:
        if param_value not in params.values():
            return params

        for k in params_to_drop:
            if k in params.keys():
                logging.info(f"remove {k} for safely run {param_value}")
                del params[k]

        return params

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return parameters of the best booster."""
        try:
            return json.loads(self.study.best_trial.system_attrs[_LGBM_PARAMS_KEY])

        except ValueError:  # TODO: parameterize this
            # Return the default score because no trials have completed.
            params = copy.deepcopy(_DEFAULT_LIGHTGBM_PARAMETERS)
            # self.lgbm_params may contain parameters given by users.
            params.update(self.lgbm_params)
            for params_to_check, v in _PARAMS_TO_CHECK:
                params = self._drop_ineffective_params(params, params_to_check, params_to_drop=v)
                
            _modify_supported_parameters(list(self.lgbm_params.values()))
            return params
