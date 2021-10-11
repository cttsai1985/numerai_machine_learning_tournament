import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn import metrics
# from sklearn import svm
from sklearn.base import BaseEstimator
from cuml import ElasticNet
from cuml import MBSGDRegressor
from cuml import svm
import scipy

from ds_utils import Utils
from ds_utils import DiagnosticUtils


def neutralize_estimator_factory(pipeline_configs: Optional[List[Dict[str, Any]]] = None) -> BaseEstimator:
    if not pipeline_configs:
        # param_grid = {"epsilon": [.1, .05, .025, .01, ], "C": [.25, .5, 1, 2, 4], "gamma": ["scale", "auto"]}
        # neutralize_estimator =
        # param_grid = {"l1_ratio": [.25, .5, .75], "alpha": [.5, 1]}
        param_grid = {"alpha": [0.0001], "l1_ratio": [.15, .25, .5]}
        neutralize_estimator = MBSGDRegressor(penalty="elasticnet", epochs=3000)
        params = {"scoring": "neg_root_mean_squared_error", "n_jobs": 1, "refit": True, "cv": 5}
        return GridSearchCV(neutralize_estimator, param_grid=param_grid, **params)

        return make_pipeline([
            StandardScaler(),
            linear_model.SGDRegressor(
                loss="squared_loss", penalty="elasticnet", max_iter=3000, alpha=0.0001, l1_ratio=0.5, random_state=0)])


class INeutralizationHelper:
    def neutralize(self, exposures: pd.DataFrame, scores: pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError()

    @property
    def has_feature_groups_(self) -> bool:
        raise NotImplementedError()


class MixinNeutralizationHelper(INeutralizationHelper):
    def __init__(
            self, feature_groups: Optional[List[str]] = None, proportion: float = 1.0,
            score_func: Optional[Callable] = None, **kwargs):
        self.feature_groups: Optional[List[str]] = feature_groups
        self.proportion: float = proportion
        self.score_func: Optional[Callable] = score_func

    @property
    def has_feature_groups_(self) -> bool:
        if self.feature_groups:
            return True

        return False

    def _neutralize(self, exposures: pd.DataFrame, scores: pd.Series, **kwargs) -> pd.Series:
        raise NotImplementedError()

    @staticmethod
    def _normalize(scores: pd.Series, **kwargs) -> pd.Series:
        return Utils.scale_uniform(scores)

    def neutralize(
            self, exposures: pd.DataFrame, scores: pd.Series, feature_groups: Optional[List[str]] = None) -> pd.Series:
        """

        :param exposures:
        :param scores:
        :param feature_groups:
        :return:
        """
        if not feature_groups:
            feature_groups = self.feature_groups

        exposures = exposures.reindex(columns=feature_groups)
        neutralized_series = self._neutralize(exposures=exposures, scores=scores)
        scores = scores - self.proportion * neutralized_series
        return scores / scores.std(ddof=0)


class NaiveNeutralizationHelper(MixinNeutralizationHelper):
    def __init__(self, feature_groups: List[str], proportion: float = 1.0, normalize: bool = False, **kwargs):
        super().__init__(feature_groups=feature_groups, proportion=proportion)
        self.normalize: bool = normalize

    def _neutralize(self, exposures: pd.DataFrame, scores: pd.Series, **kwargs) -> pd.Series:
        return DiagnosticUtils.compute_neutralize(
            exposures[self.feature_groups], scores=scores.to_frame(), proportion=self.proportion,
            normalize=self.normalize).squeeze()  # TODO: test this code

    def neutralize(
            self, exposures: pd.DataFrame, scores: pd.Series, feature_groups: Optional[List[str]] = None) -> pd.Series:
        """

        :param exposures:
        :param scores:
        :param feature_groups:
        :return:
        """
        if not feature_groups:
            feature_groups = self.feature_groups

        exposures = exposures.reindex(columns=feature_groups)
        return self._neutralize(exposures=exposures, scores=scores)


class RegNeutralizationHelper(MixinNeutralizationHelper):
    def __init__(
            self, feature_groups: List[str], proportion: float = 1.0,
            pipeline_configs: Optional[List[Dict[str, Any]]] = None, **kwargs):
        super().__init__(feature_groups=feature_groups, proportion=proportion)
        self.pipeline_configs: Optional[List[Dict[str, Any]]] = pipeline_configs
        self.debug_info: bool = False

    def _neutralize(
            self, exposures: pd.DataFrame, scores: pd.Series, **kwargs) -> pd.Series:
        neutralize_estimator = neutralize_estimator_factory(pipeline_configs=self.pipeline_configs)
        neutralized_series = neutralize_estimator.fit(exposures, scores).predict(exposures)
        if self.debug_info:
            rmse = metrics.mean_squared_error(neutralized_series, scores, squared=False)
            logging.debug(
                f"rmse: {rmse: .6f}, mean: {neutralized_series.mean(): .3f}, std: {neutralized_series.std(): .3f}")
        return neutralized_series


class _BaseMultiNeutralizationHelper(INeutralizationHelper):
    def __init__(self, neutralize_helpers: List[MixinNeutralizationHelper]):
        self.neutralize_helpers: List[MixinNeutralizationHelper] = neutralize_helpers

    @property
    def has_feature_groups_(self) -> bool:
        return False

    def neutralize(self, exposures: pd.DataFrame, scores: pd.Series, **kwargs) -> pd.Series:
        for nh in self.neutralize_helpers:
            if not nh.has_feature_groups_:
                continue

            scores = nh.neutralize(exposures=exposures, scores=scores)
            scores = nh._normalize(scores)

        return scores

    @staticmethod
    def _infer_neutralization_params(
            reference: pd.Series, quantiles: List[float], proportion_mapping: Dict[str, float]) -> List[Dict[str, Any]]:
        series = pd.qcut(reference, q=quantiles, labels=False)
        feature_groups = series.groupby(series).apply(lambda x: sorted(x.index.tolist()))
        neutralize_params = list()
        for i, feature_group in feature_groups.iteritems():
            proportion = proportion_mapping.get(i)
            if proportion is None:
                continue

            neutralize_params.append({"feature_groups": feature_group, "proportion": proportion})
        return neutralize_params


class MultiRegNeutralizationHelper(_BaseMultiNeutralizationHelper):
    @classmethod
    def from_arguments(
            cls, reference: pd.Series, quantiles: List[float], proportion_mapping: Dict[str, float],
            pipeline_configs: Optional[List[Dict[str, Any]]] = None, **kwargs):
        neutralize_helpers = list()
        for params in _BaseMultiNeutralizationHelper._infer_neutralization_params(
                reference, quantiles=quantiles, proportion_mapping=proportion_mapping):
            neutralize_helpers.append(RegNeutralizationHelper(pipeline_configs=pipeline_configs, **params))

        return cls(neutralize_helpers=neutralize_helpers)


class MultiNaiveNeutralizationHelper(_BaseMultiNeutralizationHelper):
    @classmethod
    def from_arguments(
            cls, reference: pd.Series, quantiles: List[float], proportion_mapping: Dict[str, float], **kwargs):
        neutralize_helpers = list()
        for params in _BaseMultiNeutralizationHelper._infer_neutralization_params(
                reference, quantiles=quantiles, proportion_mapping=proportion_mapping):
            neutralize_helpers.append(NaiveNeutralizationHelper(**params))

        return cls(neutralize_helpers=neutralize_helpers)
