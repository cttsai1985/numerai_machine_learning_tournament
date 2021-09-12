import sys
import os
import json
import argparse
import logging
from pathlib import Path
from glob import glob

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils.Utils import get_file_hash
from ds_utils import SolutionConfigs

ds_utils.configure_pandas_display()


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="simple utility to get file hash", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")
    args = parser.parse_args()

    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()

    _args = parse_commandline()

    root_resource_path: str = "../input/numerai_tournament_resource/"

    logging.info(f"file hash for {_args.configs}: {get_file_hash(_args.configs)}")

    configs = SolutionConfigs(
        root_resource_path=root_resource_path, configs_file_path=_args.configs, eval_data_type="training", )
    output_data_path: str = configs.output_dir_
    logging.info(f"output dir for {_args.configs}: {configs.output_dir_}")
