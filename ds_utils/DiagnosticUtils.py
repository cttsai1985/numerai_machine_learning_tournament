import logging
import numpy as np
import pandas as pd
import dask.array as da
from scipy import stats
from typing import Optional, List, Tuple, Union, Callable

from ds_utils import Utils


def payout(scores: pd.Series, lower: float = -0.25, upper: float = .25) -> float:
    """
    Payout is just the score clipped at +/-25%
    """
    return scores.clip(lower=lower, upper=upper)


def sharpe_ratio(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, np.float64]:
    return data.mean() / data.std(ddof=0)


def auto_corr_penalty(data: pd.Series, lag: int = 1) -> float:
    if not isinstance(data, pd.Series):
        raise ValueError(f"object type of data is not pd.Series ({type(data)})")

    n = data.shape[0]
    p = np.abs(data.autocorr(lag=lag))
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


def smart_sharpe(data: pd.DataFrame) -> pd.Series:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"object type of data is not pd.DataFrame ({type(data)})")

    return data.mean() / (data.std(ddof=1) * data.apply(auto_corr_penalty))


def smart_sortino_ratio(data: pd.DataFrame, target: float = .02) -> pd.Series:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"object type of data is not pd.DataFrame ({type(data)})")

    n = data.shape[0]
    xt = data - target
    return xt.mean() / (((np.sum(np.minimum(0, xt) ** 2) / (n - 1)) ** .5) * data.apply(auto_corr_penalty))


def compute_neutralize(
        exposures: pd.DataFrame, scores: pd.DataFrame, proportion: float, normalize: bool) -> pd.DataFrame:
    if normalize:
        scores = scores.apply(lambda x: stats.norm.ppf(Utils.scale_uniform(x, middle=.5)))

    _scores = scores.values.astype(np.float32)
    exposures = exposures.values.astype(np.float32)
    exposures = pd.DataFrame(
        exposures.dot(np.linalg.pinv(exposures).dot(_scores)), columns=scores.columns, index=scores.index)

    scores -= proportion * exposures
    return scores / scores.std(ddof=0)


def feature_neutral_mean(
        data: pd.DataFrame, feature_columns: List[str], proportion: float = 1.0, normalize: bool = True,
        method: str = "pearson", column_prediction: str = "prediction", column_target: str = "target",
        column_neutral: str = "neutral_score") -> pd.Series:
    data[column_neutral] = compute_neutralize(
        exposures=data[feature_columns], scores=data[[column_prediction]], proportion=proportion,
        normalize=normalize)[column_prediction]
    return Utils.scale_uniform(data[column_neutral]).corr(data[column_target], method=method)


def compute_fast_score(
        data: pd.DataFrame, columns: List[str], column_target: str, tb: Optional[int] = None):
    """

    :param data:
    :param columns:
    :param column_target:
    :param tb:
    :return:
    """
    # TODO: review later
    era_pred = data[columns].values.T.astype(np.float64)
    era_target = data[column_target].values.T.astype(np.float64)

    if tb is None:
        ccs = np.corrcoef(era_target, era_pred)[0, 1:]
    else:
        tbidx = np.argsort(era_pred, axis=1)
        tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
        ccs = np.array(
            [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)])
    return pd.Series(ccs, index=columns, name=data.name).to_frame().T


def fast_score_by_date(
        data: pd.DataFrame, columns: List[str], column_target: str, tb: Optional[int] = None,
        column_group: str = "era"):
    """

    :param data:
    :param columns:
    :param column_target:
    :param tb:
    :param column_group:
    :return:
    """
    # TODO: Need to test, This might be slow
    return data.groupby(column_group).apply(
        lambda x: compute_fast_score(x, columns=columns, column_target=column_target, tb=tb))


def neutralize_series(series: pd.Series, by: pd.Series, proportion: float = 1.0) -> pd.Series:
    """
    To neutralize any series by any other series

    :param series:
    :param by:
    :param proportion:
    :return:
    """
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack((exposures, np.array([series.mean()] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def meta_model_control(submit: pd.Series, example: pd.Series, target: pd.Series) -> float:
    series = neutralize_series(Utils.scale_uniform(submit), by=example)
    return np.cov(series, target)[0, 1] / (0.29 ** 2)


def max_draw_down(data: pd.Series, min_periods: int = 1) -> pd.Series:
    if not isinstance(data, pd.Series):
        raise ValueError(f"object type of data is not pd.Series ({type(data)})")

    daily_value: pd.Series = (data + 1.).cumprod()
    rolling_max: pd.Series = daily_value.expanding(min_periods=min_periods).max()
    return (rolling_max - daily_value) / rolling_max


def feature_exposure(
        data: pd.DataFrame, columns_feature: List[str], column_target: str = "prediction",
        method: Union[str, Callable] = "pearson", ) -> pd.Series:
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"object type of data is not pd.DataFrame ({type(data)})")

    return data[columns_feature].corrwith(other=data[column_target], method=method)
