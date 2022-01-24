import sys
import os
from pathlib import Path
import argparse
from typing import Optional, List

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

from numerai_utils import NumerAPIHelper


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh dataset")
    parser.add_argument("--update-only", action="store_true", help="download all datasets.")
    parser.add_argument("--extension", default="parquet", type=str, help="set extension")
    parser.add_argument(
        "--data-dir", default="latest_tournament_datasets", type=str, help="data folder name under the root")
    args = parser.parse_args()
    return args


if "__main__" == __name__:

    filenames: Optional[List[str]] = [
        "example_predictions.csv",
        "example_predictions.parquet",
        # "example_validation_predictions.csv",
        # "example_validation_predictions.parquet",
        "numerai_live_data.parquet",
        "numerai_tournament_data.parquet",
        # "numerai_training_data.parquet",
        # "numerai_validation_data.parquet",
        # "features.json",

    ]

    _args = parse_parameters()
    if not _args.update_only:
        filenames = None

    helper = NumerAPIHelper(data_dir=_args.data_dir)
    helper.download_latest_dataset(filenames=filenames, refresh=_args.refresh)
