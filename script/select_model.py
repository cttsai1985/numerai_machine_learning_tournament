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
    "validationMaxDrawdown": "nsmallest",

    "validationCorrPlusMmcSharpe": "nlargest",
    "validationMmcMean": "nlargest",
    "validationCorrPlusMmcSharpeDiff": "nsmallest",
    "corrWithExamplePreds": "nsmallest",
}


def compute(data: pd.Series, func: str, num: int = 3) -> pd.Series:
    return getattr(data, func)(num)


if "__main__" == __name__:
    ds_utils.initialize_logger()

    col_metric: str = "Unnamed: 0"
    root_resource_path: str = "../input/numerai_tournament_resource/"
    helper = NumerAPIHelper(root_dir_path=root_resource_path, )
    df = dd.read_csv(os.path.join(helper.result_dir_current_round_, "*.csv"), include_path_column=True).compute()
    df["path"] = df["path"].apply(lambda x: Path(x).stem)
    ret = df.set_index(["path", ]).groupby(
        col_metric).apply(lambda x: compute(x["score"], _MetricsFuncMapping.get(x.name)))
    ret = ret.reset_index("path")
    ret["target"] = list(map(lambda x: _MetricsFuncMapping.get(x), ret.index.tolist()))
    ret = ret.groupby(col_metric).apply(lambda x: x.reset_index()).reindex(columns=["target", "score", "path"])
    logging.info(f"best models:\n{ret.loc[list(_MetricsFuncMapping.keys())]}")
