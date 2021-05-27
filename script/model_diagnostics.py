import sys
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from dask import dataframe as dd
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils.Metrics import corr, payout, spearman_corr, pearson_corr


def unif(series: pd.Series) -> pd.Series:
    return (series.rank(method="first") - 0.5) / len(series)


# to neutralize any series by any other series
def neutralize_series(series: pd.Series, by: pd.Series, proportion: float = 1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def compute_single_mmc(
        x: pd.DataFrame, col_target: str = "target", col_submit: str = "yhat",
        col_example: str = "prediction") -> float:
    series = neutralize_series(unif(x[col_submit]), unif(x[col_example]))
    return np.cov(series, x[col_target])[0, 1] / (0.29 ** 2)


def _read_dataframe(filepath: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    if "id" in df.columns:
        df = df.set_index("id")

    if columns:
        return df[columns]

    return df


def compute_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return (df_corr.mean() / df_corr.std()).rename("corr sharpe")


def compute_corr_mean(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.mean().rename("corr mean")


def compute_corr_std(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.std().rename("corr std")


def compute_payout(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.apply(lambda x: payout(x)).mean().rename("payout")


def compute_max_draw_down(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    rolling_max = (df_corr + 1).cumprod().rolling(window=100, min_periods=1).max()
    daily_value = (df_corr + 1).cumprod()
    return -((rolling_max - daily_value) / rolling_max).max().rename("max draw down")


def compute_fnc(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.mean().rename("feature neutralized corr")


def _target_dataframe(prediction_filepath: str, example_filepath: str) -> pd.DataFrame:
    df_target = _read_dataframe(prediction_filepath)
    example = _read_dataframe(example_filepath)
    return df_target.join(example["prediction"], how="left")


def _compute_mmc(df: pd.DataFrame, col_group: str = "era", col_example: str = "prediction") -> pd.Series:
    return df.groupby(col_group).apply(lambda x: compute_single_mmc(x, col_example=col_example)).rename("MMC")


def compute_mmc(prediction_filepath: str, example_filepath: str) -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    return _compute_mmc(df_target, col_group="era", col_example="prediction")


def compute_corr_with_example(prediction_filepath: str, example_filepath: str) -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    corr_dict = {"Spearman": spearman_corr, "Pearson": pearson_corr}
    return df_target.groupby("era").apply(
        lambda x: pd.Series({k: v(x["prediction"], x["yhat"]) for k, v in corr_dict.items()})).mean().rename(
        "corr with example predictions")


def compute_corr_plus_mmc(
        corr_filepath: str, prediction_filepath: str, example_filepath: str,
        columns: Optional[List[str]] = ["Spearman", "Pearson"], ) -> pd.Series:
    mmc = _compute_mmc(
        _target_dataframe(prediction_filepath, example_filepath), col_group="era", col_example="prediction")
    df_corr_plus_mmc = _read_dataframe(corr_filepath, columns=columns)
    df_corr_plus_mmc["Spearman"] += mmc
    df_corr_plus_mmc["Pearson"] += mmc
    return (df_corr_plus_mmc.mean() / df_corr_plus_mmc.std()).rename("corr + mmc sharpe")


def compute_max_feature_exposure(
        prediction_filepath: str, example_filepath: str, feature_filepath: str, feature_columns_filepath: str,
        columns: Optional[List[str]] = ["Spearman", "Pearson"], column_yhat: str = "yhat",
        column_group: str = "era") -> pd.Series:
    df = dd.read_parquet(feature_file_path).set_index("id")
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    df[column_yhat] = df_target[column_yhat]
    with open(feature_columns_file_path, "r") as fp:
        columns_feature = json.load(fp)

    return df.groupby(column_group).apply(
        lambda d: pd.Series(
            {k: d[columns_feature].corrwith(d[column_yhat], method=k.lower()).abs().max() for k in columns})
    ).mean().compute().rename("max feature exposure")


if "__main__" == __name__:
    ds_utils.initialize_logger()

    root_resource_path: str = "../input/numerai_tournament_resource/"
    dataset_name: str = "latest_tournament_datasets"
    root_data_path: str = os.path.join(root_resource_path, dataset_name)
    # TODO: using configs to load
    root_prediction_path: str = "../input/numerai_tournament_resource/baseline/lightgbm_optuna_huber_6f57f220/"

    columns_corr = ["Spearman", "Pearson"]
    meta_data_path: str = "../input/numerai_tournament_resource/metadata/"
    feature_columns_file_path = os.path.join(meta_data_path, "features_numerai.json")
    example_file_path: str = os.path.join(root_data_path, "numerai_example_predictions_data.parquet")
    feature_file_path: str = os.path.join(root_data_path, "numerai_validation_data.parquet")
    score_split_file_path: str = os.path.join(root_prediction_path, "validation_score_split.parquet")
    prediction_file_path: str = os.path.join(root_prediction_path, "validation_predictions.parquet")
    feature_neutral_corr_file_path: str = os.path.join(root_prediction_path, "validation_fnc_split.parquet")
    output_file_path: str = os.path.join(root_prediction_path, "validation_model_diagnostics.{filename_extension}")

    mmc = compute_mmc(prediction_filepath=prediction_file_path, example_filepath=example_file_path)
    mmc_mean = pd.Series([mmc.mean()] * 2, index=["Spearman", "Pearson"], name="mmc mean")
    corr_with_example = compute_corr_with_example(
        prediction_filepath=prediction_file_path, example_filepath=example_file_path)

    valid_sharpe = compute_sharpe(filepath=score_split_file_path, columns=columns_corr)
    corr_plus_mmc_sharpe = compute_corr_plus_mmc(
        corr_filepath=score_split_file_path, prediction_filepath=prediction_file_path,
        example_filepath=example_file_path, columns=columns_corr)
    corr_plus_mmc_sharpe_diff = (corr_plus_mmc_sharpe - valid_sharpe).rename("corr + mmc sharpe diff")

    valid_corr = compute_corr_mean(filepath=score_split_file_path, columns=columns_corr)
    valid_std = compute_corr_std(filepath=score_split_file_path, columns=columns_corr)
    valid_payout = compute_payout(filepath=score_split_file_path, columns=columns_corr)
    max_draw_down = compute_max_draw_down(filepath=score_split_file_path, columns=columns_corr)

    valid_fnc = compute_fnc(filepath=feature_neutral_corr_file_path, columns=columns_corr)

    max_feature_exposure = compute_max_feature_exposure(
        prediction_filepath=prediction_file_path, example_filepath=example_file_path,
        feature_filepath=feature_file_path, feature_columns_filepath=feature_columns_file_path, columns=columns_corr)

    summary = pd.concat([
        valid_sharpe, valid_corr, valid_fnc, valid_payout, valid_std, max_feature_exposure, max_draw_down,
        corr_plus_mmc_sharpe, mmc_mean, corr_plus_mmc_sharpe_diff, corr_with_example, ], axis=1).T.round(6)
    logging.info(f"Validation Stats:\n{summary}")
    summary.index.name = "attr"
    summary.to_csv(output_file_path.format(filename_extension="csv"), )
    summary.to_parquet(output_file_path.format(filename_extension="parquet"), )
    # TODO: add select era
