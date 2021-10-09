import logging
from typing import List, Optional, Dict, Any, Tuple
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


def neutralize_estimator_factory(pipeline_configs: Optional[List[Dict[str, Any]]] = None) -> BaseEstimator:
    if not pipeline_configs:
        # param_grid = {"epsilon": [.1, .05, .025, .01, ], "C": [.25, .5, 1, 2, 4], "gamma": ["scale", "auto"]}
        # param_grid = {"l1_ratio": [.25, .5, .75], "alpha": [.5, 1]}
        param_grid = {"alpha": [0.0001], "l1_ratio": [.15, .25, .5]}
        return GridSearchCV(
            MBSGDRegressor(penalty="elasticnet", epochs=3000), param_grid=param_grid,
            scoring="neg_root_mean_squared_error", n_jobs=1, refit=True, cv=5)

    return make_pipeline([
        StandardScaler(),
        linear_model.SGDRegressor(
            loss="squared_loss", penalty="elasticnet", max_iter=3000, alpha=0.0001, l1_ratio=0.5, random_state=0)])


def neutralize(
        exposures: pd.DataFrame, scores: pd.Series, pipeline_configs: Optional[List[Dict[str, Any]]] = None,
        proportion: float = 1.0):
    """

    :param exposures:
    :param scores:
    :param pipeline_configs:
    :param proportion:
    :return:
    """
    neutralize_estimator = neutralize_estimator_factory(pipeline_configs=pipeline_configs)
    neutralized_series = neutralize_estimator.fit(exposures, scores).predict(exposures)
    rmse = metrics.mean_squared_error(neutralized_series, scores, squared=False)
    logging.info(f"rmse: {rmse: .6f}, mean: {neutralized_series.mean(): .3f}, std: {neutralized_series.std(): .3f}")
    scores = scores - proportion * neutralized_series
    return scores / scores.std()


def select_feature_groups_to_neutralize(
        reference: pd.Series, quantiles: Optional[List[float]] = None,
        proportion_mapping: Optional[Dict[str, float]] = None) -> List[Tuple[int, float, List[str]]]:
    series = pd.qcut(reference, q=quantiles, labels=False)
    feature_groups = series.groupby(series).apply(lambda x: x.index.tolist())
    return [(i, proportion_mapping[i], sorted(feature_group)) for i, feature_group in feature_groups.iteritems()]
