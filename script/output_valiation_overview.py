import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import pdb
from glob import glob
from dask import dataframe as dd
import shutil

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs

_MetricsFuncMapping = {
    "corr sharpe": "nlargest",
    "corr mean": "nlargest",
    "feature neutralized corr mean": "nlargest",
    "payout": "nlargest",

    "corr std": "nsmallest",
    "max feature exposure": "nsmallest",
    "max draw down": "nsmallest",

    "corr + mmc sharpe": "nlargest",
    "mmc mean": "nlargest",
    "corr + mmc sharpe diff": "nsmallest",
    "corr with example predictions": "nsmallest",

    "corr smart sharpe": "nlargest",
    "corr smart sortino ratio": "nlargest",
}


def compute(data: pd.Series, func: str, num: int = 10) -> pd.Series:
    return getattr(data, func)(num)


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-rows", type=int, default=100, help="display rows per attributes")

    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    col_metric: str = "attr"
    root_resource_path: str = "../input/numerai_tournament_resource/"

    _args = parse_commandline()
    file_pattern: str = os.path.join(
        root_resource_path, "lightgbm_optuna*", "validation_model_diagnostics.csv")

    df = dd.read_csv(os.path.join(file_pattern), include_path_column=True).compute()

    df["path"] = df["path"].apply(lambda x: Path(x).parts[-2])
    ret = df.set_index(["path", ]).groupby(
        col_metric).apply(lambda x: compute(x["Spearman"], _MetricsFuncMapping.get(x.name), num=_args.num_rows))
    ret = ret.reset_index("path")
    ret["target"] = list(map(lambda x: _MetricsFuncMapping.get(x), ret.index.tolist()))
    ret = ret.groupby(col_metric).apply(lambda x: x.reset_index()).reindex(columns=["target", "Spearman", "path"])
    ret = ret.loc[list(_MetricsFuncMapping.keys())]
    logging.info(f"best models:\n{ret}")

