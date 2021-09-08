import copy
import json
import logging
import numpy as np
import tqdm
import optuna
from typing import Optional, Callable, Any, Dict, List, Tuple, Set, Union, Generator, Iterator
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
    "learning_rate": .05,
    "lambda_l1": 1e-3,
    "lambda_l2": 1e-6,
    "num_leaves": 31,
    "max_depth": _DEFAULT_TUNER_TREE_DEPTH,  #
    "feature_fraction": .05,
    "bagging_fraction": .5,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "drop_rate": .1,  # DART specific parameters
    "skip_drop": .5,  # DART specific parameters
    "max_drop": 50,  # DART specific parameters
    "top_rate": .2,  # GOSS specific parameters
    "other_rate": .1,  # GOSS specific parameters
    "fair_c": 1.,  # fair loss specific parameters, default = 1, and > 0
    "alpha": .9,  # huber and quantile regression loss specific parameters, default = 0.9 and > 0.
    "poisson_max_delta_step": .7,  # poisson regression loss specific parameters default = 0.7 and > 0
}

_SUPPORTED_PARAMETERS: Optional[List[str]] = None
_SUPPORTED_PARAMETERS_BASE: List[str] = [
    "learning_rate",
    "lambda_l1",
    "lambda_l2",
    "num_leaves",
    "max_depth",
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
    # "poisson_max_delta_step", # poisson regression loss specific parameters default = 0.7 and > 0
]

# SUPPORTED PARAMETERS TO ADD
_ADDITIONAL_SET_OF_SUPPORTED_PARAMETERS: Dict[str, List[str]] = {
    "fair": ["fair_c"],
    "huber": ["alpha"],
    "quantile": ["alpha"],
    "poisson": ["poisson_max_delta_step"],
    "goss": ["top_rate", "other_rate"],
    "dart": ["drop_rate", "skip_drop", "max_drop"],
}
# PARAMETERS TO DROP
_NON_SUPPORTED_PARAMETERS_TO_DROP: Dict[str, List[str]] = {
    "goss": ["bagging_fraction", "bagging_freq"],
    "rf": ["learning_rate"],
}

_PARAMS_FOR_OBJECTIVE_FUNCTION = {
    "fair": "fair_c",
    "huber": "alpha",
    "quantile": "alpha",
    "poisson": "poisson_max_delta_step"
}

# SEARCH RANGE
_DEFAULT_SEARCH_RANGE: Dict[str, Dict[str, Any]] = {
    # loss function tuning
    "fair_c": {"high": 2.5, "low": 0.5, "step": 0.5, "fine_step": 0.1, "fine_grid": 2, },
    "alpha": {"high": 2.5, "low": 0.5, "step": 0.5, "fine_step": 0.1, "fine_grid": 2, },
    "poisson_max_delta_step": {"high": 1.5, "low": 0.3, "step": 0.4, "fine_step": 0.1, "fine_grid": 2, },

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


def _organize_hyper_parameters_for_tuning(params: Dict[str, Any]) -> Dict[str, Any]:
    params_base = _SUPPORTED_PARAMETERS_BASE.copy()

    for k, v in _ADDITIONAL_SET_OF_SUPPORTED_PARAMETERS.items():
        if k not in params.values():
            continue

        logging.info(f"add extra {v} hyper-parameter for {k}")
        params_base.extend(v)

    for k, v in _NON_SUPPORTED_PARAMETERS_TO_DROP.items():
        if k not in params.values():
            continue

        logging.info(f"remove {v} hyper-parameter for not allowed in {k}")
        for vv in v:
            params_base.remove(vv)
            if vv in params.keys():
                del params[vv]

    for k in params_base:
        if k in params.keys():
            continue

        params[k] = _DEFAULT_LIGHTGBM_PARAMETERS.get(k)

    globals()["_SUPPORTED_PARAMETERS"] = params_base[:]
    logging.info(f"Allowed to tuning Light GBM parameters: {_SUPPORTED_PARAMETERS}")
    return params


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
        not_supported_parameters = list(filter(lambda x: x not in _SUPPORTED_PARAMETERS, self.target_param_names))
        if not_supported_parameters:
            raise NotImplementedError(
                f'Parameter `{", ".join(not_supported_parameters)}` is not supported for tuning.')

    def _preprocess(self, trial: optuna.trial.Trial) -> None:
        if self.pbar is not None:
            self.pbar.set_description(self.pbar_fmt.format(self.step_name, self.best_score))

        if "num_leaves" in self.target_param_names:
            max_depth = self.lgbm_params.get("max_depth", _DEFAULT_TUNER_TREE_DEPTH)
            ref_tree_depth = max(_MIN_TUNER_TREE_DEPTH, max_depth)
            margin = int(np.log(ref_tree_depth ** 2))
            min_num_leaves = min_num_leave_by_depth(ref_tree_depth) + margin
            max_leaves = min(max_num_leave_by_depth(max_depth), max_num_leave_by_depth(_DEFAULT_TUNER_TREE_DEPTH - 1))
            max_num_leaves = max(filter(
                lambda x: min_num_leaves < x < max_leaves, [min_num_leaves + i * margin for i in range(1, 11)]))
            logging.info(f"num_leaves for max_depth {ref_tree_depth}: in ({min_num_leaves}, {max_num_leaves})")
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
        # self._suggest_float("fair_c", trial, round_digits=3)
        # `GridSampler` is used for huber loss tuning.
        # self._suggest_float("alpha", trial, round_digits=3)
        self._suggest_categorical("alpha", trial)
        # self._suggest_float("poisson_max_delta_step", trial, round_digits=3)
        self._suggest_categorical("poisson_max_delta_step", trial)

    def _suggest_float(
            self, param_name: str, trial: optuna.trial.Trial, round_digits: Optional[float] = None) -> None:
        if param_name not in self.target_param_names:
            return

        param = self._param_range[param_name]
        _low = param["low"]
        _high = param["high"]
        suggestion = min(trial.suggest_float(param_name, _low, _high + _EPS, log=param.get("log", False)), _high)
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
            param_distributions: Dict[str, Dict[str, Any]],
            train_set: "lgb.Dataset",
            num_boost_round: int = 1000,
            folds: Optional[Union[
                Generator[Tuple[int, int], None, None], Iterator[Tuple[int, int]], "BaseCrossValidator"]] = None,
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

        _params = _DEFAULT_SEARCH_RANGE.copy()
        if isinstance(param_distributions, dict):
            _params.update(param_distributions)

        self._param_range: Dict[str, Dict[str, Any]] = _params
        logging.info(f"parameters search range by Optuna:")
        for k, v in self._param_range.items():
            logging.info(f"parameter: {k}, search range: {v}")

        self._params_tuned: Optional[Dict[str, bool]] = None

    @staticmethod
    def _get_param_from_objective(objective_name: str) -> str:
        return _PARAMS_FOR_OBJECTIVE_FUNCTION.get(objective_name, None)

    @staticmethod
    def _create_grid_search_range(
            low: float, high: float, step: float, round_digits: Optional[int] = None) -> List[float]:
        param_range = np.linspace(low, high + _EPS, int(np.round((high - low) / step)) + 1)

        if round_digits is not None:
            param_range = np.round(param_range, round_digits)

        return param_range.tolist()

    def _tune_with_grid_template(
            self, task_name: str, param_name: str, param_search_range: Optional[List[Any]] = None):
        # three scenarios
        # 1) supply param_search_range
        # 2) a list of discrete
        # 3) a set of low, high, step
        _param_search_range: Optional[List[Any]] = None
        _discrete_lookup_key: str = "discrete"
        _range_lookup_key: List[str] = ["high", "low", "step"]
        param = self._param_range[param_name]
        go_discrete: bool = _discrete_lookup_key in param.keys()
        go_range: bool = set(_range_lookup_key).issubset(set(param.keys()))
        if param_search_range is not None:
            _param_search_range = param_search_range
            logging.info(f"create search range from predefined: {_param_search_range}")
            if go_discrete:
                self._param_range[param_name][_discrete_lookup_key] = param_search_range
            if go_range:
                pass

        elif param_search_range is None and go_discrete:
            _param_search_range = param[_discrete_lookup_key]
            logging.info(f"create search range from discrete key: {_param_search_range}")

        elif param_search_range is None and go_range:
            _param_search_range = self._create_grid_search_range(
                low=param["low"], high=param["high"], step=param["step"], round_digits=3)
            logging.info(f"create search range from high, low, step: {_param_search_range}")

        sampler = optuna.samplers.GridSampler({param_name: _param_search_range})
        self._tune_params([param_name], len(_param_search_range), sampler, task_name)

    def _refine_tune_float_with_grid_sampler(self, param_name: str, task_name: str):
        param_range = self._param_range[param_name]
        range_high: Optional[float] = param_range.get("high")
        range_low: Optional[float] = param_range.get("low")
        if not range_high and not range_low:
            if "discrete" in param_range.keys():
                _param_search_range = param_range["discrete"]
                range_high: float = max(_param_search_range)
                range_low: float = min(_param_search_range)

        fine_step: float = param_range.get("fine_step", .1)
        fine_grid: int = param_range.get("fine_grid", 2)
        fine_range: float = fine_step * fine_grid

        current_best_params = self.best_params[param_name]
        param_search_range = self._create_grid_search_range(
            current_best_params - fine_range, current_best_params + fine_range, step=fine_step, round_digits=3)
        param_search_range = list(filter(lambda val: range_high >= val >= range_low, param_search_range))
        self._tune_with_grid_template(task_name, param_name=param_name, param_search_range=param_search_range)

    def tune_learning_rate(self, task_name: str = "learning_rate", **kwargs) -> bool:
        # TODO: find the correct condition for skip
        if "rf" in self.lgbm_params.values():
            return False

        self._tune_with_grid_template(task_name=task_name, param_name="learning_rate")
        return True

    def tune_goss_parameters(self, n_trials: int = 20, task_name: str = "goss tuning") -> bool:
        _params: List[str] = ["top_rate", "other_rate"] + ["learning_rate"]
        if not set(_params).issubset(set(self.lgbm_params.keys())):
            logging.info(f"skip goss tuning")
            return False

        self._tune_with_tpe_sampler(_params, n_trials=n_trials, task_name=task_name)
        return True

    def tune_dart_parameters(self, n_trials: int = 20, task_name: str = "dart tuning") -> bool:
        _params: List[str] = ["drop_rate", "skip_drop", "max_drop"] + ["learning_rate"]
        if not set(_params).issubset(set(self.lgbm_params.keys())):
            logging.info(f"skip dart tuning")
            return False

        self._tune_with_tpe_sampler(_params, n_trials=n_trials, task_name=task_name)
        return True

    def tune_loss_function(self, **kwargs) -> bool:
        _param = self.lgbm_params.get("objective")
        obj_param_name = self._get_param_from_objective(_param)
        if obj_param_name is None:
            return False

        self._tune_with_grid_template(task_name=f"loss function tuning {obj_param_name}", param_name=obj_param_name)
        return True

    def tune_feature_fraction(
            self, task_name: str = "feature_fraction_stage1", param_name: str = "feature_fraction", **kwargs) -> None:
        self._tune_with_grid_template(task_name=f"tune {task_name}", param_name=param_name)

    def tune_feature_fraction_stage2(
            self, task_name: str = "feature_fraction_stage2", param_name: str = "feature_fraction", **kwargs) -> None:
        self._refine_tune_float_with_grid_sampler(task_name=task_name, param_name=param_name)

    def _tune_with_tpe_sampler(self, params: List[str], n_trials: int = 20, task_name: str = "num_leaves"):
        self._tune_params(params, n_trials, optuna.samplers.TPESampler(), task_name, )

    def tune_num_leaves(self, n_trials: int = 20, task_name: str = "num_leaves") -> None:
        self._tune_with_tpe_sampler(["num_leaves"], n_trials=n_trials, task_name=task_name)

    def tune_regularization_factors(self, n_trials: int = 20, task_name: str = "regularization_factors") -> None:
        self._tune_with_tpe_sampler(["lambda_l1", "lambda_l2"], n_trials=n_trials, task_name=task_name)

    def tune_min_data_in_leaf(
            self, task_name: str = "min_data_in_leaf", param_name: str = "min_child_samples", **kwargs) -> None:
        self._tune_with_grid_template(task_name=f"tune {task_name}", param_name=param_name)

    def tune_bagging(self, n_trials: int = 10, task_name: str = "bagging", **kwargs) -> bool:
        _params: List[str] = ["bagging_fraction", "bagging_freq"]

        if set(_params) != (set(self.lgbm_params.keys()) & set(_params)):
            logging.info(f"skip bagging tuning")
            return False

        self._tune_params(_params, n_trials, optuna.samplers.TPESampler(), task_name)
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

        # to initialize params to tune
        self.best_params

        # Sampling.
        self.sample_train_set()

        status = self.tune_loss_function()
        if status:
            logging.info(f"current best after tune_loss_function: {self.best_params}")

        status_goss = self.tune_goss_parameters(n_trials=25)
        if status:
            logging.info(f"current best after tune_goss_parameters: {self.best_params}")

        status_dart = self.tune_dart_parameters(n_trials=25)
        if status_dart:
            logging.info(f"current best after tune_dart_parameters: {self.best_params}")

        if not (status_dart or status_goss):
            self.tune_learning_rate()
            logging.info(f"current best after tune_learning_rate: {self.best_params}")

        self.tune_feature_fraction()
        logging.info(f"current best after tune_feature_fraction: {self.best_params}")
        self.tune_num_leaves(n_trials=20)
        logging.info(f"current best after tune_num_leaves: {self.best_params}")

        status = self.tune_bagging(n_trials=20)
        if status:
            logging.info(f"current best after tune_bagging: {self.best_params}")

        self.tune_feature_fraction_stage2()
        logging.info(f"current best after tune_feature_fraction_stage2: {self.best_params}")

        status = self.tune_goss_parameters(n_trials=15, task_name="fine-tuned goss boosting")
        if status:
            logging.info(f"current best after tune_goss_parameters 2nd time: {self.best_params}")

        status = self.tune_dart_parameters(n_trials=15, task_name="fine-tuned dart boosting")
        if status:
            logging.info(f"current best after tune_dart_parameters 2nd time: {self.best_params}")

        self.tune_regularization_factors(n_trials=20)
        logging.info(f"current best after tune_regularization_factors: {self.best_params}")
        self.tune_min_data_in_leaf()
        logging.info(f"current best after tune_min_data_in_leaf: {self.best_params}")
        # TODO: fine tune some params later

    @staticmethod
    def _drop_ineffective_params(params: Dict[str, Any], param_value: Any, params_to_drop: List[str]) -> Dict[str, Any]:
        if param_value not in params.values():
            return params

        for k in params_to_drop:
            if k in params.keys():
                logging.info(f"remove {k} for safely run {param_value}")
                del params[k]

        for k, v in params.items():
            logging.info(f"default value for {k}: {v} in search")
        return params

    @property
    def best_params(self) -> Dict[str, Any]:
        """Return parameters of the best booster."""
        try:
            return json.loads(self.study.best_trial.system_attrs[_LGBM_PARAMS_KEY])

        except ValueError:  # TODO: parameterize this
            # Return the default score because no trials have completed.

            params = _organize_hyper_parameters_for_tuning(self.lgbm_params)
            self._params_tuned = {k: False for k in params.keys()}
            return params
