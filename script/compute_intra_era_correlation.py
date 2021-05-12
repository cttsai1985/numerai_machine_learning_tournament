import sys
import os
import json
import re
import multiprocessing as mp
import logging
from functools import partial
from itertools import product
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple
import pandas as pd
from dask import dataframe as dd
from dask.distributed import LocalCluster
from dask.distributed import Client

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils


def map_func(x: str, reg: re.Pattern, output_format: str, digits: int = 5) -> str:
    try:
        ret = reg.search(x).groupdict()
    except:
        return x

    ret["number"] = ret["number"].zfill(digits)
    return output_format.format(**ret)


def normalize_era(
        index: List[str], pattern: str = "^(?P<prefix>[A-z]+)(?P<number>[0-9]+)",
        output_format: str = "{prefix}{number}", digits: int = 5, n_jobs: Optional[int] = None) -> List[str]:
    reg = re.compile(pattern)
    f = partial(map_func, reg=reg, output_format=output_format, digits=digits)
    with mp.Pool(n_jobs if n_jobs else mp.cpu_count()) as p:
        ret = p.map(f, index)

    return ret


def compute(data_filepath: str, columns: List[str], groupby_col: str = "era", method: str = "spearman",
            output_filepath: str = "./df.parquet", n_jobs: Optional[int] = None, refresh: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(data_filepath, columns=["id", groupby_col] + columns).set_index("id")
    df.rename(columns={i: i.split("_")[1] for i in columns}, inplace=True)

    ret: pd.DataFrame = pd.DataFrame()
    if os.path.exists(output_filepath) and os.path.isfile(output_filepath):
        ret = pd.read_parquet(output_filepath)
        logging.info(f"read {ret.shape} from {output_filepath}")

    current_all_era = df[groupby_col].unique().tolist()
    current_all_era_normalize = normalize_era(current_all_era)
    if not refresh and (set(normalize_era(ret.index.tolist())) == set(current_all_era_normalize)):
        logging.info(f"result file exists and skips: {output_filepath}")
        return ret

    df[groupby_col] = df[groupby_col].map({k: v for k, v in zip(current_all_era, current_all_era_normalize)})
    mask = ~df[groupby_col].isin(ret.index)
    if not mask.any():
        return ret

    logging.info(f"compute {mask.sum()} rows for {method}")
    df = dd.from_pandas(df.loc[mask], chunksize=50000)
    df = df.groupby(groupby_col).apply(lambda x: x.corr(method=method)).compute(n_workers=n_jobs, processes=False)
    df = df.unstack()
    df.index = normalize_era(df.index.tolist())
    df.columns = df.columns.tolist()
    df = df.reindex(columns=list(filter(lambda x: len(set(x)) != 1, df.columns.tolist())))
    df.columns = list(map(lambda x: "_".join(x), df.columns.tolist()))
    if not ret.empty:
        df = pd.concat([ret, df], axis=0)

    df.sort_index(inplace=True)
    logging.info(f"write into {output_filepath}: {df.shape}")
    df.to_parquet(output_filepath)
    return df


if "__main__" == __name__:
    ds_utils.initialize_logger()

    # cluster = LocalCluster(n_workers=12, processes=False)
    # client = Client(cluster)

    output_data_path: str = "../input/numerai_tournament_resource/data_correlation/"
    Path(output_data_path).mkdir(parents=True, exist_ok=True)

    meta_data_path: str = "../input/numerai_tournament_resource/metadata/"
    feature_columns_set = os.path.join(meta_data_path, "features_numera*.json")

    root_data_path: str = "../input/numerai_tournament_resource/latest_tournament_datasets/"
    data_file_types: List[str] = ["train", "valid", "test", "live"]
    data_filenames: List[str] = [
        "numerai_training_data.parquet", "numerai_validation_data.parquet", "numerai_test_data.parquet",
        "numerai_live_data.parquet"]

    df_pairs: List[Tuple[str, str]] = [
        (i, j) for i, j in zip(data_file_types, map(lambda x: os.path.join(root_data_path, x), data_filenames))]

    column_pairs = [(Path(i).stem.split("_")[1], json.load(open(i, "r"))) for i in glob(feature_columns_set)]
    method_pairs = [("spearman", "spearman"), ("pearson", "pearson"), ("kendall", "kendall")]

    for seq in product(df_pairs, column_pairs, method_pairs):
        obj = list(zip(*seq))
        names = obj[0]
        items = obj[1]
        opt_refresh: bool = True if names[0] == "live" else False

        filename = os.path.join(output_data_path, "_".join([names[0], names[1], "correl", names[2], ])) + ".parquet"
        compute(
            data_filepath=items[0], columns=items[1], method=items[2], groupby_col="era", output_filepath=filename,
            n_jobs=-1, refresh=opt_refresh)
