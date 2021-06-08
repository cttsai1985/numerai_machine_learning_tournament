import sys
import os
import json
import argparse
import logging
from pathlib import Path
from numerapi import NumerAPI

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs
from numerai_utils import NumerAPIHelper


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")
    parser.add_argument("--numerapiPublicID", type=str, default="", help="numerapi public ID")
    parser.add_argument("--numerapiSecret", type=str, default="", help="numerapi secret")
    parser.add_argument("--model-name", type=str, default="cttsai", help="")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()

    root_resource_path: str = "../input/numerai_tournament_resource/"
    dataset_name: str = "latest_tournament_datasets"
    _args = parse_commandline()
    logging.info(f"{_args.numerapiSecret}, {_args.numerapiPublicID}")
    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    helper = NumerAPIHelper(
        root_dir_path=root_resource_path,
        api=NumerAPI(secret_key=_args.numerapiSecret, public_id=_args.numerapiPublicID))
    helper.evaluate_online_predictions(model_name=_args.model_name, dir_path=configs.output_dir_)
