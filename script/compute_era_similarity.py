import sys
import os
import json
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

# import cupy as cp
# import cudf
# import dask_cudf
# Initialize UCX for high-speed transport of CUDA arrays
from dask_cuda import LocalCUDACluster
from cuml.metrics.pairwise_distances import pairwise_distances

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils


# intra-list average local distance (ILALD)
def compute_mean_ilald(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).mean() if x.name in x.index else x.mean()).mean()


# intra-list average local distance (ILALD)
def compute_median_ilald(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).mean() if x.name in x.index else x.mean()).median()


# intra-list average local distance (ILALD)
def compute_std_ilald(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).mean() if x.name in x.index else x.mean()).std()


# intra-list average local distance (ILALD)
def compute_skew_ilald(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).mean() if x.name in x.index else x.mean()).skew()


# intra-list minimal local distance (ILMLD)
def compute_mean_ilmld(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).min() if x.name in x.index else x.min()).mean()


# intra-list minimal local distance (ILMLD)
def compute_std_ilmld(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).min() if x.name in x.index else x.min()).std()


# intra-list minimal local distance (ILMLD)
def compute_median_ilmld(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).min() if x.name in x.index else x.min()).median()


# intra-list minimal local distance (ILMLD)
def compute_skew_ilmld(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).min() if x.name in x.index else x.min()).skew()


# intra-list median local distance (ILMeLD)
def compute_median_ilmeld(df: pd.DataFrame) -> float:
    return df.apply(lambda x: x.drop(x.name).median() if x.name in x.index else x.median()).median()


def compute_distance(x: pd.DataFrame, y: pd.DataFrame, func: Callable) -> pd.DataFrame:
    return pd.DataFrame(func(x.values, y.values), columns=y.index, index=x.index)


def compute_distance_metrics(
        x: pd.DataFrame, y: pd.DataFrame, distance_func: Callable, metrics: Dict[str, Callable]) -> Dict[str, Any]:
    df = compute_distance(x.reindex(columns=y.columns), y, func=distance_func)
    return {k: v(df) for k, v in metrics.items()}


def compute(
        data_filepath: str, target_filepath: str, columns: List[str], dist_func: Callable, metrics: Dict[str, Callable],
        groupby_col: str = "era", output_filepath: str = "./df.parquet", n_jobs: Optional[int] = None,
        refresh: bool = False):
    df2 = pd.read_parquet(target_filepath).set_index("id")
    max_era = df2[groupby_col].max()
    df2 = df2.reindex(columns=columns)

    output_filename = output_filepath.format(max_era=max_era)
    if os.path.exists(output_filename) and not refresh:
        df = pd.read_parquet(output_filename)
        logging.info(f"found and not refresh {output_filename}: {df.shape}\n{df.describe()}")
        return df

    df1 = dd.read_parquet(
        data_filepath, columns=["id", groupby_col] + df2.columns.tolist(), chuchsize=50000).set_index("id")
    logging.info(f"{max_era} columns ({len(columns)}): data: ({df1.shape[0]}) * target({df2.shape[0]})")
    df = df1.groupby(groupby_col).apply(
        lambda x: pd.Series(compute_distance_metrics(x, df2, dist_func, metrics))).compute(
        n_workers=n_jobs, processes=False)

    logging.info(f"write into {output_filename}: {df.shape}\n{df.describe()}")
    df.to_parquet(output_filename)
    return df


if "__main__" == __name__:
    ds_utils.initialize_logger()

    # Create a Dask single-node CUDA cluster w/ one worker per device
    # cluster = LocalCUDACluster(protocol="ucx", enable_tcp_over_ucx=True, enable_nvlink=True, enable_infiniband=False)
    # cluster = LocalCluster(n_workers=12, processes=False)
    # client = Client(cluster)

    root_data_path: str = "../input/numerai_tournament_resource/latest_tournament_datasets/"
    meta_data_path: str = "../input/numerai_tournament_resource/metadata/"
    output_data_path: str = "../input/numerai_tournament_resource/data_distance/"

    feature_columns_set = os.path.join(meta_data_path, "features_nu*.json")
    Path(output_data_path).mkdir(parents=True, exist_ok=True)

    # item[0]
    data_file_types: List[str] = ["train", "valid", "test"]
    data_filenames: List[str] = [
        "numerai_training_data.parquet", "numerai_validation_data.parquet", "numerai_test_data.parquet"]
    df1_pairs: List[Tuple[str, str]] = [
        (i, j) for i, j in zip(data_file_types, map(lambda x: os.path.join(root_data_path, x), data_filenames))]

    # df_to_split = pd.read_parquet(os.path.join(root_data_path, "numerai_validation_data.parquet"))
    # target_file_types: List[str] = df_to_split["era"].unique().tolist()
    # target_filenames: List[str] = ["numerai_validation_data_{era}.parquet".format(era=i) for i in target_file_types]
    # df_to_split.groupby("era").apply(lambda x: x.to_parquet(
    #      os.path.join(root_data_path, "numerai_validation_data_{era}.parquet".format(era=x["era"].max()))))

    # item[1]
    target_file_types: List[str] = ["live", "maxTestEra"]
    target_filenames: List[str] = ["numerai_live_data.parquet", "numerai_max_test_era_data.parquet"]
    df2_pairs: List[Tuple[str, str]] = [
        (i, j) for i, j in zip(target_file_types, map(lambda x: os.path.join(root_data_path, x), target_filenames))]

    # item[2]
    column_pairs = [(Path(i).stem.split("_")[1], json.load(open(i, "r"))) for i in glob(feature_columns_set)]
    distance_types: List[str] = ["cosine", "manhattan", "euclidean"]
    distance_funcs: List[Tuple[str, Callable]] = [(i, partial(pairwise_distances, metric=i)) for i in distance_types]

    _metrics = {
        "mean_ilald": compute_mean_ilald,  # "std_ilald": compute_std_ilald,
        "mean_ilmld": compute_mean_ilmld,  # "std_ilmld": compute_std_ilmld,
        "median_ilald": compute_median_ilald, "median_ilmld": compute_median_ilmld,
        "skew_ilald": compute_skew_ilald, "skew_ilmld": compute_skew_ilmld,
        "median_ilmeld": compute_median_ilmeld
    }
    for seq in product(df1_pairs, df2_pairs, column_pairs, distance_funcs):
        obj = list(zip(*seq))
        names = obj[0]
        items = obj[1]

        option_refresh: bool = False
        if names[1] == "live":
            option_refresh = True

        filename = os.path.join(
            output_data_path, "_".join(["{max_era}", names[0], names[2], "distance", names[3], ])) + ".parquet"
        compute(
            data_filepath=items[0], target_filepath=items[1], columns=items[2], dist_func=items[3],
            metrics={"_".join([names[3], k]): v for k, v in _metrics.items()}, groupby_col="era",
            output_filepath=filename, n_jobs=-1, refresh=option_refresh)

    # import pdb; pdb.set_trace()
