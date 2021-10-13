import sys
import os
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import FilenameTemplate as ft
from ds_utils import SolutionConfigs
from ds_utils import DiagnosticUtils


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")
    parser.add_argument("--refresh", action="store_true", help="refresh")

    args = parser.parse_args()
    return args


def compute_corr(
        feature_file_path: str, output_corr_file_path: str, column_group: str, columns_feature: List[str],
        column_target: str, refresh: bool = False) -> pd.DataFrame:
    if refresh or (not (os.path.exists(output_corr_file_path) and os.path.isfile(output_corr_file_path))):
        df = pd.read_parquet(feature_file_path)
        df_corr = df.groupby(column_group).apply(
            lambda x: x[columns_feature].corrwith(x[column_target], method="pearson"))
        df_corr.to_parquet(output_corr_file_path)
        logging.info(f"save correlation {df_corr.shape} to {output_corr_file_path}")

    else:
        df_corr = pd.read_parquet(output_corr_file_path)
        logging.info(f"load correlation {df_corr.shape} from {output_corr_file_path}")
    return df_corr


def compute_corr_stats(df_corr: pd.DataFrame, output_feature_analysis_filepath: str, refresh: bool = False):
    if refresh or (not (os.path.exists(
            output_feature_analysis_filepath) and os.path.isfile(output_feature_analysis_filepath))):
        df = pd.concat([
            df_corr.mean().rename("corrMean"),
            df_corr.std().rename("corrStd"),
            df_corr.median().rename("corrMedian"),
            DiagnosticUtils.sharpe_ratio(df_corr).rename("corrSharpe"),
            DiagnosticUtils.smart_sharpe(df_corr).rename("corrSmartSharpe"),
            DiagnosticUtils.smart_sortino_ratio(df_corr).rename("corrSmartSortinoRatio")], axis=1, sort=False)
        df.to_parquet(output_feature_analysis_filepath)
    else:
        df = pd.read_parquet(output_feature_analysis_filepath)
        logging.info(f"load feature analysis {df.shape} from {output_feature_analysis_filepath}")

    return df


def compute(
        root_data_path: str, eval_data_type: str = "validation", column_group: str = "era",
        column_target: str = "target", refresh: bool = False, **kwargs) -> pd.DataFrame:
    # tournament files
    feature_columns_file_path = os.path.join(configs.meta_data_dir, ft.default_feature_collection_filename)
    with open(feature_columns_file_path, "r") as fp:
        columns_feature = json.load(fp)

    feature_file_path: str = os.path.join(
        root_data_path, ft.numerai_data_filename_template.format(eval_type=eval_data_type))
    output_corr_file_path: str = os.path.join(
        root_data_path, ft.feature_corr_filename_template.format(eval_type=eval_data_type, target=column_target))
    df_corr = compute_corr(
        feature_file_path=feature_file_path, output_corr_file_path=output_corr_file_path, column_group=column_group,
        columns_feature=columns_feature, column_target=column_target, refresh=refresh)

    output_feature_analysis_file_path: str = os.path.join(
        root_data_path, ft.feature_analysis_filename_template.format(eval_type=eval_data_type, target=column_target))
    df = compute_corr_stats(
        df_corr, output_feature_analysis_filepath=output_feature_analysis_file_path, refresh=refresh)

    for i, row in df.abs().apply(lambda x: ','.join(x.nsmallest(200).index)).iteritems():
        logging.info(f"{i}: {row}")

    return df


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    _args = parse_commandline()
    configs = SolutionConfigs(root_resource_path=ft.root_resource_path, configs_file_path=_args.configs)
    _ = compute(
        root_data_path=ft.default_data_dir, eval_data_type="validation", column_group="era",
        column_target=configs.column_target, refresh=_args.refresh)
    _ = compute(
        root_data_path=ft.default_data_dir, eval_data_type="training", column_group="era",
        column_target=configs.column_target, refresh=_args.refresh)
