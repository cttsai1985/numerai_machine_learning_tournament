import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Any
import pdb
from glob import glob
from dask import dataframe as dd
import shutil

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import FilenameTemplate as ft
from numerai_utils import NumerAPIHelper

_MetricsFuncMapping = {
    "corrSharpe": "nlargest",
    "corrMean": "nlargest",
    "corrSmartSharpe": "nlargest",
    "corrSmartSortinoRatio": "nlargest",
    "payout": "nlargest",

    "corrStd": "nsmallest",
    "maxFeatureExposure": "nsmallest",
    "maxDrawDown": "nlargest",

    "corrPlusMmcSharpe": "nlargest",
    "mmcMean": "nlargest",
    "corrPlusMmcSharpeDiff": "nlargest",  # due to minus value
    "examplePredsCorr": "nsmallest",

    "corrSmartSharpe": "nlargest",
    "corrSmartSortinoRatio": "nlargest",
}


def compute(data: pd.Series, func: str, num: int = 10) -> pd.Series:
    return getattr(data, func)(num)


def parse_commandline() -> argparse.Namespace:
    default_output_pattern: str = "*"
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-rows", type=int, default=500, help="display rows per attributes")
    parser.add_argument(
        "--corr-type", type=str, default="Pearson", help="metric for correlation")
    parser.add_argument(
        "--output-dir", type=str, default="./", help="output dir")
    parser.add_argument(
        "--output-pattern", type=str, default=default_output_pattern, help="destination to move output dirs")
    parser.add_argument(
        "--corr-cut-off", type=float, default=.72, help="max corr with example predictions")
    parser.add_argument(
        "--max-draw-down", type=float, default=-100., help="max draw down cut off")
    parser.add_argument(
        "--max-corr-std", type=float, default=1., help="max corr std cut off")
    parser.add_argument(
        "--min-mmc", type=float, default=0., help="max corr std cut off")
    parser.add_argument(
        "--max-corr-mmc-sharpe", type=float, default=0., help="max corr std cut off")
    parser.add_argument(
        "--min-corr-sharpe", type=float, default=.5, help="min sharpe ratio cut off")
    parser.add_argument("--use-filter", action="store_true", help="use filter to select output")
    args = parser.parse_args()
    return args


def filter_model_by_performance(
        data: pd.DataFrame, metric_selections: List[Tuple[str, Any]], column_score: str,
        column: str = "path") -> List[str]:
    allow_list: List[str] = list()
    for _metric, _cutoff in metric_selections:
        _cmp = _MetricsFuncMapping[_metric]
        sub = data.loc[_metric,]
        if _cmp == "nsmallest":
            mask = sub[column_score] <= _cutoff
        if _cmp == "nlargest":
            mask = sub[column_score] >= _cutoff

        logging.info(f"{_metric} cutoff={_cutoff:.3f}: {mask.sum()}")
        allow_list.append(sub[column].loc[mask].tolist())

    all_allow_list = set(data[column].tolist())
    allow_list = list(map(lambda x: set(x), allow_list))
    for i in allow_list:
        all_allow_list = all_allow_list & i
        logging.info(f"count: {len(all_allow_list)}")

    return all_allow_list


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    _column_objective: str = "objective"
    _col_score: str = "score"
    _column_path: str = "path"
    col_metric: str = "attr"
    root_resource_path: str = ft.root_resource_path

    _args = parse_commandline()

    # read files
    file_pattern: str = os.path.join(root_resource_path, _args.output_pattern, "validation_model_diagnostics.csv")
    df = dd.read_csv(os.path.join(file_pattern), include_path_column=True).compute()

    df = df.loc[df[col_metric].isin(_MetricsFuncMapping.keys())]
    df[_column_path] = df[_column_path].apply(lambda x: Path(x).parts[-2])
    _metrics_names = df[col_metric].unique()
    _metric_func_mapping = {k: v for k, v in _MetricsFuncMapping.items() if k in _metrics_names}
    ret = df.set_index([_column_path, ]).groupby(
        col_metric).apply(lambda x: compute(x[_col_score], _metric_func_mapping.get(x.name), num=_args.num_rows))
    ret = ret.reset_index(_column_path)
    ret[_column_objective] = list(map(lambda x: _metric_func_mapping.get(x), ret.index.tolist()))
    ret = ret.groupby(col_metric).apply(
        lambda x: x.reset_index()).reindex(columns=[_column_objective, _col_score, _column_path])
    ret = ret.loc[list(_metric_func_mapping.keys())]

    # multiple sub
    _allow_list: List[str] = list()
    _metric_selections: List[Tuple[str, float]] = [
        ("corrSharpe", _args.min_corr_sharpe),
        ("corrStd", _args.max_corr_std),
        ("maxDrawDown", _args.max_draw_down),
        ("mmcMean", _args.min_mmc),
        ("corrPlusMmcSharpeDiff", _args.max_corr_mmc_sharpe),
        ("examplePredsCorr", _args.corr_cut_off)
    ]

    if _args.use_filter:
        _all_allow_list = filter_model_by_performance(
            data=ret, metric_selections=_metric_selections, column_score=_col_score, column=_column_path)

        logging.info(f"rows: {len(_allow_list)}, after filtered: {len(_all_allow_list)}")
        ret = ret.loc[ret[_column_path].isin(_all_allow_list)]

    helper = NumerAPIHelper(root_dir_path=root_resource_path, )
    Path(helper.result_dir_current_round_).mkdir(parents=True, exist_ok=True)
    logging.info(f"best models:\n{ret}")
    ret.to_csv(os.path.join(helper.result_dir_current_round_, "summary.csv"), index=True)
    output_filepath = os.path.join(helper.result_dir_current_round_, "summary.txt")
    with open(output_filepath, "w") as f:
        f.write(ret.to_string())
        logging.info(f"write the ranking results to: {output_filepath}")
