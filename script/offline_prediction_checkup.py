import sys
import os
import json
import argparse
import logging
import pandas as pd
from pathlib import Path

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs


def compute(root_data_path: str, root_prediction_path: str) -> pd.DataFrame:
    data_file_path: str = os.path.join(root_data_path, "numerai_tournament_data.parquet")
    prediction_file_path: str = os.path.join(root_prediction_path, "tournament_predictions.parquet")
    output_file_path: str = os.path.join(
        root_prediction_path, "_".join(["prediction", "diagnostics.{filename_extension}"]))

    df = pd.read_parquet(data_file_path, columns=["id", "data_type", "era"]).set_index("id")
    df_preds = pd.read_parquet(prediction_file_path).set_index("id")
    df["prediction"] = df_preds["prediction"]
    group_df = df.groupby(["data_type", "era"])["prediction"].describe().groupby(level=0)
    df_mean = group_df.mean()
    df_mean["stats"] = "mean"
    df_std = group_df.std().dropna()
    df_std["stats"] = "std"
    summary = pd.concat([df_mean, df_std])

    logging.info(f"prediction stats on :\n{summary}")
    summary.to_csv(output_file_path.format(filename_extension="csv"), )
    summary.to_parquet(output_file_path.format(filename_extension="parquet"), )
    return summary


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")

    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    root_resource_path: str = "../input/numerai_tournament_resource/"
    dataset_name: str = "latest_tournament_datasets"
    _args = parse_commandline()
    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    _root_data_path = os.path.join(root_resource_path, dataset_name)
    compute(root_data_path=_root_data_path, root_prediction_path=configs.output_dir_, )
