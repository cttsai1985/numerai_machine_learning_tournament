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
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs
from ds_utils import Utils
from ds_utils import DiagnosticUtils
from ds_utils import FilenameTemplate as ft


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
    return df_target.join(example["prediction"].rename("example_prediction"), how="left")


def compute_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.sharpe_ratio(df_corr).rename("corr sharpe")


def compute_smart_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.smart_sharpe(df_corr).rename("corr smart sharpe")


def compute_smart_sortino_ratio(filepath: str, columns: Optional[List[str]] = None, target: float = .02) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.smart_sortino_ratio(df_corr, target=target).rename("corr smart sortino ratio")


def compute_corr_mean(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.mean().rename("corr mean")


def compute_corr_std(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.std(ddof=0).rename("corr std")


def compute_payout(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.apply(lambda x: DiagnosticUtils.payout(x)).mean().rename("payout")


def compute_max_draw_down(
        filepath: str, columns: Optional[List[str]] = None, min_periods: int = 1) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    draw_down = df_corr.apply(lambda x: DiagnosticUtils.max_draw_down(x, min_periods=min_periods))
    return -(draw_down.max()).rename("max draw down")


def compute_fnc(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.mean().rename("feature neutralized corr mean")


def compute_feature_neutral_mean(
        prediction_filepath: str, feature_filepath: str, feature_columns_filepath: str,
        columns: Optional[List[str]] = None, column_group: str = "era", column_prediction: str = "prediction",
        column_target: str = "target", proportion: float = 1., normalize: bool = True) -> pd.Series:
    df = _read_dataframe(feature_filepath)
    df_target = _read_dataframe(prediction_filepath)
    df[column_prediction] = df_target[column_prediction]
    with open(feature_columns_filepath, "r") as fp:
        columns_feature = json.load(fp)

    df["neutral_sub"] = df.groupby(column_group).apply(
        lambda x: DiagnosticUtils.compute_neutralize(
            x, target_columns=[column_prediction], neutralizers=columns_feature, proportion=proportion,
            normalize=normalize))

    results = dict()
    groupby_df = df.groupby(column_group)
    for method in columns:
        results[method] = groupby_df.apply(lambda x: Utils.scale_uniform(x["neutral_sub"]).corr(
            other=(x[column_target]), method=method.lower()))
    return pd.DataFrame(results).mean().rename("feature neutralized corr mean")


def _compute_meta_model_control(
        df: pd.DataFrame, column_group: str = "era", column_example: str = "example_prediction",
        column_prediction: str = "prediction", column_target: str = "target") -> pd.Series:
    df.dropna(inplace=True)
    return df.groupby(column_group).apply(
        lambda x: DiagnosticUtils.meta_model_control(
            submit=x[column_prediction], example=x[column_example], target=x[column_target])).rename(
        "meta_model_control")


def compute_mmc_mean(
        prediction_filepath: str, example_filepath: str, columns: Optional[List[str]] = None,
        column_group: str = "era", column_example: str = "example_prediction", column_prediction: str = "prediction",
        column_target: str = "target") -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    mmc = _compute_meta_model_control(
        df=df_target, column_group=column_group, column_example=column_example, column_prediction=column_prediction,
        column_target=column_target)
    return pd.Series([mmc.mean()] * len(columns), index=columns, name="mmc mean")


def compute_corr_with_example(
        prediction_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, column_group: str = "era",
        column_example: str = "example_prediction", column_prediction: str = "prediction") -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)

    groupby_df = df_target.groupby(column_group)
    results = dict()
    for method in columns:
        results[method] = groupby_df.apply(
            lambda x: Utils.scale_uniform(x[column_example]).corr(
                other=Utils.scale_uniform(x[column_prediction]), method=method.lower()))
    return pd.DataFrame(results).mean().rename("corr with example predictions")


def compute_corr_plus_mmc(
        corr_filepath: str, prediction_filepath: str, example_filepath: str, columns: Optional[List[str]] = None,
        column_group: str = "era", column_example: str = "example_prediction", column_prediction: str = "prediction",
        column_target: str = "target", ) -> pd.Series:
    df_target = _target_dataframe(prediction_filepath, example_filepath)
    df_mmc = _compute_meta_model_control(
        df=df_target, column_group=column_group, column_example=column_example, column_prediction=column_prediction,
        column_target=column_target)
    df_corr = _read_dataframe(corr_filepath, columns=columns)
    for col in df_corr.columns:
        df_corr[col] = df_corr[col] + df_mmc
    return (df_corr.mean() / df_corr.std()).rename("corr + mmc sharpe")


def compute_corr_plus_mmc_sharpe_diff(
        corr_filepath: str, prediction_filepath: str, example_filepath: str, columns: Optional[List[str]] = None,
        column_group: str = "era", column_example: str = "example_prediction", column_prediction: str = "prediction",
        column_target: str = "target", ) -> pd.Series:
    corr_plus_mmc_sharpe = compute_corr_plus_mmc(
        corr_filepath=corr_filepath, prediction_filepath=prediction_filepath, example_filepath=example_filepath,
        columns=columns, column_group=column_group, column_example=column_example, column_prediction=column_prediction,
        column_target=column_target)
    valid_sharpe = compute_sharpe(filepath=corr_filepath, columns=columns)
    return (corr_plus_mmc_sharpe - valid_sharpe).rename("corr + mmc sharpe diff")


def max_feature_exposure(
        data: pd.DataFrame, method: str, columns_feature: List[str],
        column_prediction: str = "prediction", ) -> pd.Series:
    return data[columns_feature].corrwith(other=data[column_prediction], method=method).abs().max()


def compute_max_feature_exposure(
        prediction_filepath: str, feature_filepath: str, feature_columns_filepath: str,
        columns: Optional[List[str]] = None, column_group: str = "era",
        column_prediction: str = "prediction") -> pd.Series:
    df = _read_dataframe(feature_filepath)
    df_target = _read_dataframe(prediction_filepath)
    df[column_prediction] = df_target[column_prediction]
    with open(feature_columns_filepath, "r") as fp:
        columns_feature = json.load(fp)

    groupby_df = df.groupby(column_group)
    results = dict()
    for method in columns:
        results[method] = groupby_df.apply(
            lambda d: max_feature_exposure(
                d, method=method.lower(), columns_feature=columns_feature, column_prediction=column_prediction))
    return pd.DataFrame(results).mean().rename("max feature exposure")


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")

    args = parser.parse_args()
    return args


def compute(
        root_data_path: str, root_prediction_path: str, eval_data_type: str = "validation",
        columns_corr: Optional[List[str]] = None, column_group: str = "era", column_example: str = "example_prediction",
        column_prediction: str = "prediction", column_target: str = "target",
        allow_func_list: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    if eval_data_type not in ["training", "validation"]:
        logging.info(f"eval_data_type is not allowed: {eval_data_type}")
        return pd.DataFrame()

    result_type = {"training": "cross_val", "validation": "validation"}.get(eval_data_type)

    # produced files
    score_split_file_path: str = os.path.join(
        root_prediction_path, ft.score_split_filename_template.format(eval_type=result_type))
    prediction_file_path: str = os.path.join(
        root_prediction_path, ft.predictions_parquet_filename_template.format(eval_type=result_type))
    feature_neutral_corr_file_path: str = os.path.join(
        root_prediction_path, f"{result_type}_fnc_split.parquet")

    # tournament files
    feature_columns_file_path = os.path.join(configs.meta_data_dir, "features_numerai.json")
    example_file_path: str = os.path.join(root_data_path, ft.example_validation_predictions_parquet_filename)
    feature_file_path: str = os.path.join(
        root_data_path, ft.numerai_data_filename_template.format(eval_type=eval_data_type))
    output_file_path: str = os.path.join(root_prediction_path, ft.model_diagnostics_filename_template)

    file_paths: List[str] = [
        score_split_file_path, prediction_file_path,
        # feature_neutral_corr_file_path
        feature_columns_file_path, example_file_path, feature_file_path
    ]
    file_status: List[bool] = list(map(lambda x: os.path.exists(x), file_paths))
    if not all(file_status):
        logging.info(f"skip computing {eval_data_type}")
        for file_path, status in zip(file_paths, file_status):
            if not status:
                logging.info(f"missing result file: {file_path}")

        return pd.DataFrame()

    # configure computing tasks
    func_list: List[Tuple[str, Callable, Dict[str, Any]]] = [
        ("valid_sharpe", compute_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_corr", compute_corr_mean, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_feature_neutral_corr", compute_feature_neutral_mean,
         dict(prediction_filepath=prediction_file_path, feature_filepath=feature_file_path,
              feature_columns_filepath=feature_columns_file_path, columns=columns_corr, column_group=column_group,
              column_prediction=column_prediction, column_target=column_target, proportion=1., normalize=True)),
        ("valid_fnc", compute_fnc, dict(filepath=feature_neutral_corr_file_path, target_columns=columns_corr)),
        ("valid_smart_sharpe", compute_smart_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_smart_sortino_ratio", compute_smart_sortino_ratio,
         dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_payout", compute_payout, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("valid_std", compute_corr_std, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("max_feature_exposure", compute_max_feature_exposure,
         dict(prediction_filepath=prediction_file_path, feature_filepath=feature_file_path,
              feature_columns_filepath=feature_columns_file_path, columns=columns_corr, column_group=column_group,
              column_prediction=column_prediction, )),
        ("max_draw_down", compute_max_draw_down, dict(filepath=score_split_file_path, columns=columns_corr)),
    ]

    func_list_with_mmc: List[Tuple[str, Callable, Dict[str, Any]]] = [
        ("corr_plus_mmc_sharpe", compute_corr_plus_mmc, dict(
            corr_filepath=score_split_file_path, prediction_filepath=prediction_file_path,
            example_filepath=example_file_path, columns=columns_corr, column_group=column_group,
            column_example=column_example, column_prediction=column_prediction, column_target=column_target)),
        ("mmc_mean", compute_mmc_mean, dict(
            prediction_filepath=prediction_file_path, example_filepath=example_file_path, columns=columns_corr,
            column_group=column_group, column_example=column_example, column_prediction=column_prediction,
            column_target=column_target)),
        ("corr_plus_mmc_sharpe_diff", compute_corr_plus_mmc_sharpe_diff, dict(
            corr_filepath=score_split_file_path, prediction_filepath=prediction_file_path,
            example_filepath=example_file_path, columns=columns_corr, column_group=column_group,
            column_example=column_example, column_prediction=column_prediction, column_target=column_target)),
        ("corr_with_example", compute_corr_with_example, dict(
            prediction_filepath=prediction_file_path, example_filepath=example_file_path, columns=columns_corr,
            column_group=column_group, column_example=column_example, column_prediction=column_prediction, )),
    ]

    if eval_data_type == "validation":
        func_list += func_list_with_mmc

    # doing the real computing here
    func_list = list(filter(lambda x: x[0] in allow_func_list, func_list))
    ret_collect = list()
    for i in func_list:
        name, func, params = i
        logging.info(f"compute {name}")
        locals()[name] = func(**params)
        ret_collect.append(locals()[name])

    summary = pd.concat(list(filter(lambda x: ~x.empty, ret_collect)), axis=1).T.round(4)
    summary.index.name = "attr"
    logging.info(f"stats on {eval_data_type}:\n{summary}")
    summary.to_csv(output_file_path.format(eval_type=result_type, filename_extension="csv"), )
    summary.to_parquet(output_file_path.format(eval_type=result_type, filename_extension="parquet"), )
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

    _allow_func_list: List[str] = [
        "valid_sharpe",
        "valid_corr",
        # "valid_feature_neutral_corr",
        # "valid_fnc",
        "valid_smart_sharpe",
        "valid_smart_sortino_ratio",
        "valid_payout",
        "valid_std",
        # "max_feature_exposure",
        "max_draw_down",
        "corr_plus_mmc_sharpe",
        "mmc_mean",
        "corr_plus_mmc_sharpe_diff",
        "corr_with_example"
    ]

    _column_target = configs.column_target
    _column_target: str = "target"

    cross_val_summary = compute(
        root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, eval_data_type="training",
        columns_corr=_columns_corr, column_group="era", column_example="example_prediction",
        column_prediction="prediction", column_target=_column_target, allow_func_list=_allow_func_list)
    validation_summary = compute(
        root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, eval_data_type="validation",
        columns_corr=_columns_corr, column_group="era", column_example="example_prediction",
        column_prediction="prediction", column_target=_column_target, allow_func_list=_allow_func_list)

    if not (cross_val_summary.empty or validation_summary.empty):
        diff_summary = (validation_summary - cross_val_summary).dropna().reindex(index=cross_val_summary.index)
        _output_file_path: str = os.path.join(
            configs.output_dir_, "_".join(["diff", "model", "diagnostics.{filename_extension}"]))
        diff_summary.to_csv(_output_file_path.format(filename_extension="csv"), )
        diff_summary.to_parquet(_output_file_path.format(filename_extension="parquet"), )
        logging.info(f"difference stats for over-fitting analysis:\n{diff_summary}")
