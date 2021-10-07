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


def compute_feature_neutral_mean(
        prediction_filepath: str, feature_filepath: str, feature_columns_filepath: str,
        columns: Optional[List[str]] = None, column_group: str = "era", column_prediction: str = "prediction",
        column_target: str = "target", proportion: float = 1., normalize: bool = True) -> pd.Series:
    # TODO: remove this later
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


def compute_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.sharpe_ratio(df_corr).rename("corrSharpe")


def compute_smart_sharpe(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.smart_sharpe(df_corr).rename("corrSmartSharpe")


def compute_smart_sortino_ratio(filepath: str, columns: Optional[List[str]] = None, target: float = .02) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return DiagnosticUtils.smart_sortino_ratio(df_corr, target=target).rename("corrSmartSortinoRatio")


def compute_corr_mean(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.mean().rename("corrMean")


def compute_corr_std(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.std(ddof=0).rename("corrStd")


def compute_payout(filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    return df_corr.apply(lambda x: DiagnosticUtils.payout(x)).mean().rename("payout")


def compute_max_draw_down(
        filepath: str, columns: Optional[List[str]] = None, min_periods: int = 1) -> pd.Series:
    df_corr = _read_dataframe(filepath, columns=columns)
    draw_down = df_corr.apply(lambda x: DiagnosticUtils.max_draw_down(x, min_periods=min_periods))
    return -(draw_down.max()).rename("maxDrawDown")


def compute_mmc_mean(example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_mmc = _read_dataframe(example_filepath, columns=["metaModelControl"])
    df_mmc = df_mmc.mean().rename("mmcMean")
    df_mmc.index = columns
    return df_mmc


def compute_mmc_std(example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_mmc = _read_dataframe(example_filepath, columns=["metaModelControl"])
    df_mmc = df_mmc.mean().rename("mmcStd")
    df_mmc.index = columns
    return df_mmc


def compute_corr_with_example(example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_example_preds_corr = _read_dataframe(example_filepath, columns=["examplePredsCorr"])
    example_preds_corr = df_example_preds_corr.mean().rename("examplePredsCorr")
    example_preds_corr.index = columns
    return example_preds_corr


def compute_corr_plus_mmc(
        corr_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = _read_dataframe(corr_filepath, columns=columns)
    df_mmc = _read_dataframe(example_filepath, columns=["metaModelControl"])

    df_corr = df_corr.squeeze() + df_mmc.squeeze()
    return df_corr.rename("corrPlusMmc")


def compute_corr_plus_mmc_mean(
        corr_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = compute_corr_plus_mmc(
        corr_filepath=corr_filepath, example_filepath=example_filepath, columns=columns)
    return pd.Series(df_corr.mean(), index=columns, name="corrPlusMmcMean")


def compute_corr_plus_mmc_std(
        corr_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = compute_corr_plus_mmc(
        corr_filepath=corr_filepath, example_filepath=example_filepath, columns=columns)
    return pd.Series(df_corr.std(), index=columns, name="corrPlusMmcStd")


def compute_corr_plus_mmc_sharpe(
        corr_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    df_corr = compute_corr_plus_mmc(
        corr_filepath=corr_filepath, example_filepath=example_filepath, columns=columns)
    return pd.Series(DiagnosticUtils.sharpe_ratio(df_corr), index=columns, name="corrPlusMmcSharpe")


def compute_corr_plus_mmc_sharpe_diff(
        corr_filepath: str, example_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    corr_plus_mmc_sharpe = compute_corr_plus_mmc_sharpe(
        corr_filepath=corr_filepath, example_filepath=example_filepath, columns=columns)
    valid_sharpe = compute_sharpe(filepath=corr_filepath, columns=columns)
    return (corr_plus_mmc_sharpe - valid_sharpe).rename("corrPlusMmcSharpeDiff")


def compute_max_feature_exposure(feature_exposure_filepath: str, columns: Optional[List[str]] = None, ) -> pd.Series:
    data = _read_dataframe(feature_exposure_filepath)
    return pd.Series(data.abs().max(axis=1).mean(), index=columns, name="maxFeatureExposure")


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
        column_prediction: str = "prediction", column_target: str = "target", tb_num: Optional[int] = None,
        allow_func_list: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
    if eval_data_type not in ["training", "validation"]:
        logging.info(f"eval_data_type is not allowed: {eval_data_type}")
        return pd.DataFrame()

    result_type = {"training": "cross_val", "validation": "validation"}.get(eval_data_type)

    # produced files
    score_split_file_path: str = os.path.join(
        root_prediction_path, ft.score_split_filename_template.format(eval_type=result_type))
    feature_exposure_file_path: str = os.path.join(
        root_prediction_path, ft.feature_exposure_filename_template.format(eval_type=result_type))
    example_analytics_file_path: str = os.path.join(
        root_prediction_path, ft.example_analytics_filename_template.format(eval_type=result_type))

    output_file_path: str = os.path.join(root_prediction_path, ft.model_diagnostics_filename_template)
    if tb_num:
        score_split_file_path: str = os.path.join(
            root_prediction_path, ft.score_tb_split_filename_template.format(eval_type=result_type, tb_num=tb_num))
        feature_exposure_file_path: str = os.path.join(
            root_prediction_path,
            ft.feature_exposure_tb_filename_template.format(eval_type=result_type, tb_num=tb_num))
        example_analytics_file_path: str = os.path.join(
            root_prediction_path,
            ft.example_analytics_tb_filename_template.format(eval_type=result_type, tb_num=tb_num))

        output_file_path: str = os.path.join(root_prediction_path, ft.model_diagnostics_tb_filename_template)

    file_paths: List[str] = [
        score_split_file_path, feature_exposure_file_path, example_analytics_file_path,
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
        ("validSharpe", compute_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("validCorr", compute_corr_mean, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("validSmartSharpe", compute_smart_sharpe, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("validSmartSortinoRatio", compute_smart_sortino_ratio,
         dict(filepath=score_split_file_path, columns=columns_corr)),
        ("validPayout", compute_payout, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("validStd", compute_corr_std, dict(filepath=score_split_file_path, columns=columns_corr)),
        ("maxFeatureExposure", compute_max_feature_exposure, dict(
            feature_exposure_filepath=feature_exposure_file_path, columns=columns_corr)),
        ("maxDrawDown", compute_max_draw_down, dict(filepath=score_split_file_path, columns=columns_corr)), ]

    func_list_with_mmc: List[Tuple[str, Callable, Dict[str, Any]]] = [
        ("corrPlusMmcSharpe", compute_corr_plus_mmc_sharpe, dict(
            corr_filepath=score_split_file_path, example_filepath=example_analytics_file_path, columns=columns_corr, )),
        ("validMmcMean", compute_mmc_mean, dict(example_filepath=example_analytics_file_path, columns=columns_corr, )),
        ("validCorrPlusMmcSharpeDiff", compute_corr_plus_mmc_sharpe_diff, dict(
            corr_filepath=score_split_file_path, example_filepath=example_analytics_file_path, columns=columns_corr, )),
        ("validCorrPlusMmcMean", compute_corr_plus_mmc_mean, dict(
            corr_filepath=score_split_file_path, example_filepath=example_analytics_file_path, columns=columns_corr, )),
        ("validCorrPlusMmcStd", compute_corr_plus_mmc_std, dict(
            corr_filepath=score_split_file_path, example_filepath=example_analytics_file_path, columns=columns_corr, )),
        ("examplePredsCorr", compute_corr_with_example, dict(
            example_filepath=example_analytics_file_path, columns=columns_corr, )),
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
        #
        ret_collect.append(locals()[name])

    summary = pd.concat(list(filter(lambda x: ~x.empty, ret_collect)), axis=1).T
    summary.index.name = "attr"
    summary.columns = ["score"]
    logging.info(f"stats on {eval_data_type}:\n{summary.round(4)}")
    if not tb_num:
        summary.to_csv(output_file_path.format(eval_type=result_type, filename_extension="csv"), )
        summary.to_parquet(output_file_path.format(eval_type=result_type, filename_extension="parquet"), )
    else:
        summary.to_csv(output_file_path.format(eval_type=result_type, filename_extension="csv", tb_num=tb_num), )
        summary.to_parquet(
            output_file_path.format(eval_type=result_type, filename_extension="parquet", tb_num=tb_num), )
    # TODO: add select era
    return summary


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    root_resource_path: str = ft.root_resource_path
    _args = parse_commandline()
    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    _root_data_path = ft.default_data_dir
    _columns_corr: List[str] = ["Pearson"]  # ["Spearman", "Pearson"]
    _allow_func_list: List[str] = [
        # profit
        "validSharpe",
        "validCorr",
        "validSmartSharpe",
        "validSmartSortinoRatio",

        # risk
        "validPayout",
        "validStd",
        "maxFeatureExposure",
        "maxDrawDown",

        # meta model control
        "corrPlusMmcSharpe",
        "validMmcMean",
        "validCorrPlusMmcSharpeDiff",
        "examplePredsCorr",
        "validCorrPlusMmcMean",
        "validCorrPlusMmcStd",
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
    validation_tb_summary = compute(
        root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, eval_data_type="validation",
        columns_corr=_columns_corr, column_group="era", column_example="example_prediction", tb_num=200,
        column_prediction="prediction", column_target=_column_target, allow_func_list=_allow_func_list)

    if not (validation_tb_summary.empty or validation_summary.empty):
        diff_summary = (validation_tb_summary - validation_summary).dropna().reindex(index=validation_summary.index)
        _output_file_path: str = os.path.join(
            configs.output_dir_, "_".join(["diff", "model", "diagnostics.{filename_extension}"]))
        diff_summary.to_csv(_output_file_path.format(filename_extension="csv"), )
        diff_summary.to_parquet(_output_file_path.format(filename_extension="parquet"), )
        logging.info(f"difference stats for over-fitting analysis:\n{diff_summary}")
