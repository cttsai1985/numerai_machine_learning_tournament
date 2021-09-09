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

_MetricsFuncMapping = {
    "corr sharpe": "nlargest",
    "corr mean": "nlargest",
    "feature neutralized corr mean": "nlargest",
    "payout": "nlargest",

    "corr std": "nsmallest",
    "max feature exposure": "nsmallest",
    "max draw down": "nlargest",

    "corr + mmc sharpe": "nlargest",
    "mmc mean": "nlargest",
    "corr + mmc sharpe diff": "nsmallest",  # due to minus value
    "corr with example predictions": "nsmallest",

    "corr smart sharpe": "nlargest",
    "corr smart sortino ratio": "nlargest",
}


def compute(data: pd.Series, func: str, num: int = 10) -> pd.Series:
    return getattr(data, func)(num)


def parse_commandline() -> argparse.Namespace:
    default_output_pattern: str = "lightgbm_optuna*"
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-rows", type=int, default=500, help="display rows per attributes")
    parser.add_argument(
        "--output-dir", type=str, default="./", help="output dir")
    parser.add_argument(
        "--output-pattern", type=str, default=default_output_pattern, help="destination to move output dirs")
    parser.add_argument(
        "--corr-cut-off", type=float, default=.9625, help="max corr with example predictions")
    parser.add_argument(
        "--max-draw-down", type=float, default=-.045, help="max draw down cut off")
    parser.add_argument(
        "--max-corr-std", type=float, default=.0285, help="max corr std cut off")
    parser.add_argument(
        "--min-corr-sharpe", type=float, default=.825, help="min sharpe ratio cut off")
    parser.add_argument("--use-filter", action="store_true", help="use filter to select output")
    args = parser.parse_args()
    return args


def filter_model_by_performance(
        data: pd.DataFrame, metric_selections: List[Tuple[str, Any]], column: str = "path") -> List[str]:
    allow_list: List[str] = list()
    for _metric, _cutoff in metric_selections:
        _cmp = _MetricsFuncMapping[_metric]
        sub = data.loc[_metric,]
        if _cmp == "nsmallest":
            mask = sub[_column_score] <= _cutoff
        if _cmp == "nlargest":
            mask = sub[_column_score] >= _cutoff

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

    _column_target: str = "target"
    _column_path: str = "path"
    _column_score: str = "Spearman"
    col_metric: str = "attr"
    root_resource_path: str = "../input/numerai_tournament_resource/"

    _args = parse_commandline()
    file_pattern: str = os.path.join(
        root_resource_path, _args.output_pattern, "validation_model_diagnostics.csv")

    df = dd.read_csv(os.path.join(file_pattern), include_path_column=True).compute()

    df[_column_path] = df[_column_path].apply(lambda x: Path(x).parts[-2])
    ret = df.set_index([_column_path, ]).groupby(
        col_metric).apply(lambda x: compute(x[_column_score], _MetricsFuncMapping.get(x.name), num=_args.num_rows))
    ret = ret.reset_index(_column_path)
    ret[_column_target] = list(map(lambda x: _MetricsFuncMapping.get(x), ret.index.tolist()))
    ret = ret.groupby(col_metric).apply(
        lambda x: x.reset_index()).reindex(columns=[_column_target, _column_score, _column_path])
    ret = ret.loc[list(_MetricsFuncMapping.keys())]

    # multiple sub
    allow_list: List[str] = list()
    _metric_selections: List[Tuple[str, float]] = [
        ("corr sharpe", _args.min_corr_sharpe), ("corr std", _args.max_corr_std),
        ("max draw down", _args.max_draw_down), ("corr with example predictions", _args.corr_cut_off)]

    if _args.use_filter:
        _all_allow_list = filter_model_by_performance(
            data=ret, metric_selections=_metric_selections, column=_column_path)

        ret = ret.loc[ret[_column_path].isin(_all_allow_list)]

    logging.info(f"best models:\n{ret}")
    output_filepath = os.path.join(_args.output_dir, "summary.txt")
    with open(output_filepath, "w") as f:
        f.write(ret.to_string())
        logging.info(f"write the ranking results to: {output_filepath}")
