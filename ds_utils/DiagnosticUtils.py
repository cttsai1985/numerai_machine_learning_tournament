import logging
import numpy as np
import pandas as pd
import scipy
from typing import Optional, List, Tuple, Union

from ds_utils import Utils


def payout(scores: pd.Series, lower: float = -0.25, upper: float = .25) -> float:
    """
    Payout is just the score cliped at +/-25%
    """
    return scores.clip(lower=lower, upper=upper)


def sharpe_ratio(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    return data.mean() / data.std(ddof=0)


def auto_corr_penalty(x: pd.Series, lag: int = 1) -> float:
    n = x.shape[0]
    p = np.abs(x.autocorr(lag=lag))
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


def smart_sharpe(data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    return data.mean() / (data.std(ddof=1) * data.apply(auto_corr_penalty))


def smart_sortino_ratio(data: pd.DataFrame, target: float = .02) -> pd.Series:
    xt = data - target
    return xt.mean() / (
            ((np.sum(np.minimum(0, xt) ** 2) / (xt.shape[0] - 1)) ** .5) * data.apply(auto_corr_penalty))


def compute_neutralize(
        data: pd.DataFrame, columns: List[str], neutralizers: List[str], proportion: float, normalize: bool):
    scores = data[columns].values
    if normalize:
        scores2 = []
        for x in scores.T:
            x = (scipy.stats.rankdata(x, method='ordinal') - .5) / len(x)
            x = scipy.stats.norm.ppf(x)
            scores2.append(x)
        scores = np.array(scores2).T

    exposures = data[neutralizers].values
    scores -= proportion * exposures.dot(
        np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

    scores /= scores.std(ddof=0)
    return pd.Series(scores, index=data.index)


def neutralize_by_era(
        data: pd.DataFrame, columns: List[str], neutralizers: List[str], proportion: float = 1.0,
        normalize: bool = True, column_group: str = "era"):
    if not neutralizers:
        raise ValueError()

    return data.groupby(column_group).apply(lambda x: compute_neutralize(
        x, columns=columns, neutralizers=neutralizers, proportion=proportion, normalize=normalize))


def get_feature_neutral_mean(
        data: pd.DataFrame, feature_cols: List[str], column_target: str, column_prediction: str,
        column_group: str = "era") -> float:
    # TODO: not test
    data["neutral_sub"] = neutralize_by_era(
        data, [column_prediction], feature_cols, column_group=column_group)[column_prediction]
    scores = data.groupby(column_group).apply(
        lambda x: (Utils.scale_uniform(x["neutral_sub"]).corr(x[column_target]))).mean()
    return np.mean(scores)


def compute_fast_score(
        data: pd.DataFrame, columns: List[str], column_target: str, tb: Optional[int] = None):
    era_pred = data[columns].values.T.astype(np.float64)
    era_target = data[column_target].values.T.astype(np.float64)

    if tb is None:
        ccs = np.corrcoef(era_target, era_pred)[0, 1:]
    else:
        tbidx = np.argsort(era_pred, axis=1)
        tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
        ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
        ccs = np.array(ccs)

    return pd.Series(ccs, index=columns, name=data.name).to_frame().T


def fast_score_by_date(
        data: pd.DataFrame, columns: List[str], column_target: str, tb: Optional[int] = None,
        column_group: str = "era"):
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
    series = neutralize_series(Utils.scale_uniform(submit), example)
    return np.cov(series, target)[0, 1] / (0.29 ** 2)
