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
    parser.add_argument("--refresh", action="store_true", help="force upload predictions")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    root_resource_path: str = "../input/numerai_tournament_resource/"
    _args = parse_commandline()
    logging.info(f"NumerAI credentials: public_id={_args.numerapiPublicID}, secret={_args.numerapiSecret}")

    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    napi = NumerAPI(secret_key=_args.numerapiSecret, public_id=_args.numerapiPublicID)
    helper = NumerAPIHelper(root_dir_path=root_resource_path, api=napi)
    helper.submit_prediction(model_name=_args.model_name, dir_path=configs.output_dir_, refresh=_args.refresh)
    # helper.evaluate_online_diagnostics(model_name=_args.model_name, dir_path=configs.output_dir_, refresh=_args.refresh)