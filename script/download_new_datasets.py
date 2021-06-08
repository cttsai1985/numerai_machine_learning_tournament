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
    parser.add_argument("--extension", default="parquet", help="set extension")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    _args = parse_parameters()
    helper = NumerAPIHelper()

    valid_data_types: Optional[List[str]] = [
        "live", "test", "max_test_era", "tournament", "tournament_ids", "example_predictions"]
    if not _args.update_only:
        valid_data_types = None

    helper.download_latest_dataset(extension=_args.extension, valid_data_types=valid_data_types, refresh=_args.refresh)
