import sys
import os
import argparse
import logging
import pandas as pd
from dask import dataframe as dd
from pathlib import Path
from typing import Callable

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from numerai_utils import NumerAPIHelper

_MetricsFuncMapping = {
    "validationSharpe": "nlargest",
    "validationCorrelation": "nlargest",
    "validationFeatureNeutralMean": "nlargest",

    "validationStd": "nsmallest",
    "validationMaxFeatureExposure": "nsmallest",
    "validationMaxDrawdown": "nlargest",

    "validationCorrPlusMmcSharpe": "nlargest",
    "validationMmcMean": "nlargest",
    "validationCorrPlusMmcSharpeDiff": "nlargest",  # minus value
    "corrWithExamplePreds": "nsmallest",
}


def compute(data: pd.Series, func: str, num: int = 10) -> pd.Series:
    return getattr(data, func)(num)


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-rows", type=int, default=25, help="display rows per attributes")
    parser.add_argument("--num-round", type=int, default=None, help="the specific round instead of the live round.")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    _args = parse_commandline()

    col_metric: str = "attr"
    root_resource_path: str = "../input/numerai_tournament_resource/"
    helper = NumerAPIHelper(root_dir_path=root_resource_path, )

    file_pattern = helper.result_dir_current_round_
    if _args.num_round is not None:
        file_pattern = os.path.join(
            root_resource_path, "live_rounds", f"numerai_tournament_round_{_args.num_round:04d}")

    df = dd.read_csv(os.path.join(file_pattern, "*.csv"), include_path_column=True).compute()
    df["path"] = df["path"].apply(lambda x: Path(x).stem)
    ret = df.set_index(["path", ]).groupby(
        col_metric).apply(lambda x: compute(x["score"], _MetricsFuncMapping.get(x.name), num=_args.num_rows))
    ret = ret.reset_index("path")
    ret["target"] = list(map(lambda x: _MetricsFuncMapping.get(x), ret.index.tolist()))
    ret = ret.groupby(col_metric).apply(lambda x: x.reset_index()).reindex(columns=["target", "score", "path"])
    ret = ret.loc[list(_MetricsFuncMapping.keys())]
    logging.info(f"best models:\n{ret}")

    output_filepath = os.path.join(file_pattern, "summary.txt")
    with open(output_filepath, "w") as f:
        f.write(ret.to_string())
        logging.info(f"write the ranking results to: {output_filepath}")
