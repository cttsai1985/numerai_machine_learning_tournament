import sys
import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
from dask import dataframe as dd
from dask import array as da
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs
from ds_utils.Utils import scale_uniform, neutralize_series, auto_corr_penalty
from ds_utils.Metrics import payout, scale_uniform, spearman_corr, pearson_corr


def compute_single_mmc(
        x: pd.DataFrame, col_target: str = "target", col_submit: str = "yhat",
        col_example: str = "prediction") -> float:
    series = neutralize_series(scale_uniform(x[col_submit]), scale_uniform(x[col_example]))
    return np.cov(series, x[col_target])[0, 1] / (0.29 ** 2)


def _read_dataframe(
        filepath: str, columns: Optional[List[str]] = None,
        use_dask: bool = False) -> Union[pd.DataFrame, dd.DataFrame]:
    df = pd.read_parquet(filepath)
    if "id" in df.columns:
        df = df.set_index("id")

    if columns:
        df = df[columns]

    if use_dask:
        return dd.from_pandas(df, chunksize=10000)

    return df


def _target_dataframe(
        prediction_filepath: str, example_filepath: str, use_dask: bool = False) -> Union[pd.DataFrame, dd.DataFrame]:
    df_target = _read_dataframe(prediction_filepath, use_dask=use_dask)
    example = _read_dataframe(example_filepath, use_dask=use_dask)
    return df_target.join(example["prediction"], how="left")


def compute_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return (df_corr.mean() / df_corr.std()).rename("corr sharpe")


def compute_smart_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return (df_corr.mean() / (df_corr.std(ddof=1) * df_corr.apply(auto_corr_penalty))).rename("corr smart sharpe")


def compute_smart_sortino_ratio(filepath: str, columns: Optional[List[str]] = None, target: float = .02) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    xt = df_corr - target
    ret = xt.mean() / (((np.sum(np.minimum(0, xt) ** 2) / (xt.shape[0] - 1)) ** .5) * df_corr.apply(auto_corr_penalty))
    return ret.rename("corr smart sortino ratio")


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
    return df_corr.mean().rename("feature neutralized corr mean")


def _compute_mmc(df: pd.DataFrame, col_group: str = "era", col_example: str = "prediction") -> pd.Series:
    return df.groupby(col_group).apply(lambda x: compute_single_mmc(x, col_example=col_example)).rename("MMC")


def compute_mmc(prediction_filepath: str, example_filepath: str) -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    return _compute_mmc(df_target, col_group="era", col_example="prediction")


def comupte_mmc_mean(
        prediction_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    mmc = compute_mmc(prediction_filepath=prediction_filepath, example_filepath=example_filepath)
    return pd.Series([mmc.mean()] * len(columns), index=columns, name="mmc mean")


def compute_corr_with_example(prediction_filepath: str, example_filepath: str) -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    corr_dict = {"Spearman": spearman_corr, "Pearson": pearson_corr}
    return df_target.groupby("era").apply(
        lambda x: pd.Series({k: v(x["prediction"], x["yhat"]) for k, v in corr_dict.items()})).mean().rename(
        "corr with example predictions")


def compute_corr_plus_mmc(
        corr_filepath: str, prediction_filepath: str, example_filepath: str,
        columns: Optional[List[str]] = None, ) -> pd.Series:
    mmc = _compute_mmc(
        _target_dataframe(prediction_filepath, example_filepath), col_group="era", col_example="prediction")
    df_corr_plus_mmc = _read_dataframe(corr_filepath, columns=columns)
    for i in columns:
        df_corr_plus_mmc[i] += mmc
    return (df_corr_plus_mmc.mean() / df_corr_plus_mmc.std()).rename("corr + mmc sharpe")


def compute_corr_plus_mmc_sharpe_diff(
        corr_filepath: str, prediction_filepath: str, example_filepath: str,
        columns: Optional[List[str]] = None, ) -> pd.Series:
    corr_plus_mmc_sharpe = compute_corr_plus_mmc(
        corr_filepath=corr_filepath, prediction_filepath=prediction_filepath, example_filepath=example_filepath,
        columns=columns)
    valid_sharpe = compute_sharpe(filepath=corr_filepath, columns=columns)
    return (corr_plus_mmc_sharpe - valid_sharpe).rename("corr + mmc sharpe diff")


def max_feature_exposure(
        data: pd.DataFrame, columns_corr: List[str], columns_feature: List[str],
        column_yhat: str = "yhat", ) -> pd.Series:
    return pd.Series(
        {k: data[columns_feature].corrwith(data[column_yhat], method=k.lower()).abs().max() for k in columns_corr})


def compute_max_feature_exposure(
        prediction_filepath: str, feature_filepath: str, feature_columns_filepath: str,
        columns: Optional[List[str]] = None, column_yhat: str = "yhat", column_group: str = "era") -> pd.Series:
    df = _read_dataframe(feature_filepath)
    df_target = _read_dataframe(prediction_filepath)
    df[column_yhat] = df_target[column_yhat]
    with open(feature_columns_filepath, "r") as fp:
        columns_feature = json.load(fp)

    return df.groupby(column_group).apply(
        lambda d: max_feature_exposure(d, columns, columns_feature, column_yhat)).mean().rename("max feature exposure")


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")

    args = parser.parse_args()
    return args


def compute(
        root_data_path: str, root_prediction_path: str, eval_data_type: str = "validation",
        columns_corr: Optional[List[str]] = None) -> pd.DataFrame:
    if eval_data_type not in ["training", "validation"]:
        logging.info(f"eval_data_type is not allowed: {eval_data_type}")
        return pd.DataFrame()

    result_type = {"training": "cross_val", "validation": "validation"}.get(eval_data_type)

    score_split_file_path: str = os.path.join(root_prediction_path, f"{result_type}_score_split.parquet")
    prediction_file_path: str = os.path.join(root_prediction_path, f"{result_type}_predictions.parquet")
    feature_neutral_corr_file_path: str = os.path.join(root_prediction_path, f"{result_type}_fnc_split.parquet")

    file_paths: List[str] = [score_split_file_path, prediction_file_path, feature_neutral_corr_file_path]
    file_status: List[bool] = list(map(lambda x: os.path.exists(x), file_paths))
    if not all(file_status):
        logging.info(f"skip computing {eval_data_type}")
        for file_path, status in zip(file_paths, file_status):
            if not status:
                logging.info(f"missing result file: {file_path}")

        return pd.DataFrame()

    feature_columns_file_path = os.path.join(configs.meta_data_dir, "features_numerai.json")
    example_file_path: str = os.path.join(root_data_path, "numerai_example_predictions_data.parquet")
    feature_file_path: str = os.path.join(root_data_path, f"numerai_{eval_data_type}_data.parquet")
    output_file_path: str = os.path.join(
        root_prediction_path, "_".join([f"{result_type}", "model", "diagnostics.{filename_extension}"]))

    # configure computing tasks
    func_list: List[Tuple[str, Callable, Dict[str, Any]]] = [
        ("valid_sharpe", compute_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_corr", compute_corr_mean, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_fnc", compute_fnc, dict(filepath=feature_neutral_corr_file_path, columns=columns_corr)),
        ("valid_smart_sharpe", compute_smart_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_smart_sortino_ratio", compute_smart_sortino_ratio,
         dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_payout", compute_payout, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_std", compute_corr_std, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("max_feature_exposure", compute_max_feature_exposure, dict(
            prediction_filepath=prediction_file_path, feature_filepath=feature_file_path,
            feature_columns_filepath=feature_columns_file_path, columns=columns_corr)),
        ("max_draw_down", compute_max_draw_down, dict(filepath=score_split_file_path, columns=columns_corr)),
    ]

    func_list_with_mmc: List[Tuple[str, Callable, Dict[str, Any]]] = [
        ("corr_plus_mmc_sharpe", compute_corr_plus_mmc, dict(
            corr_filepath=score_split_file_path, prediction_filepath=prediction_file_path,
            example_filepath=example_file_path, columns=columns_corr)),
        ("mmc_mean", comupte_mmc_mean, dict(
            prediction_filepath=prediction_file_path, example_filepath=example_file_path, columns=columns_corr)),
        ("corr_plus_mmc_sharpe_diff", compute_corr_plus_mmc_sharpe_diff, dict(
            corr_filepath=score_split_file_path, prediction_filepath=prediction_file_path,
            example_filepath=example_file_path, columns=columns_corr)),
        ("corr_with_example", compute_corr_with_example, dict(
            prediction_filepath=prediction_file_path, example_filepath=example_file_path)),
    ]

    if eval_data_type == "validation":
        func_list += func_list_with_mmc

    # doing the real computing here
    ret_collect = list()
    for i in func_list:
        name, func, params = i
        logging.info(f"compute {name}")
        locals()[name] = func(**params)
        ret_collect.append(locals()[name])

    summary = pd.concat(ret_collect, axis=1).T.round(6)
    summary.index.name = "attr"
    logging.info(f"stats on {eval_data_type}:\n{summary}")
    summary.to_csv(output_file_path.format(filename_extension="csv"), )
    summary.to_parquet(output_file_path.format(filename_extension="parquet"), )
    # TODO: add select era
    return summary


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    root_resource_path: str = "../input/numerai_tournament_resource/"
    dataset_name: str = "latest_tournament_datasets"
    _args = parse_commandline()
    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    _root_data_path = os.path.join(root_resource_path, dataset_name)
    _columns_corr: List[str] = ["Spearman", "Pearson"]

    cross_val_summary = compute(
        root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, eval_data_type="training",
        columns_corr=_columns_corr)
    validation_summary = compute(
        root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, eval_data_type="validation",
        columns_corr=_columns_corr)

    if not (cross_val_summary.empty or validation_summary.empty):
        diff_summary = (validation_summary - cross_val_summary).dropna().reindex(index=cross_val_summary.index)
        _output_file_path: str = os.path.join(
            configs.output_dir_, "_".join(["diff", "model", "diagnostics.{filename_extension}"]))
        diff_summary.to_csv(_output_file_path.format(filename_extension="csv"), )
        diff_summary.to_parquet(_output_file_path.format(filename_extension="parquet"), )
        logging.info(f"difference stats for over-fitting analysis:\n{diff_summary}")
