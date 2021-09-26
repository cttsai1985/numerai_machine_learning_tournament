import sys
import os
import argparse
import logging
from pathlib import Path

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import FilenameTemplate as ft
from ds_utils import SolutionConfigs
from numerai_utils import NumerAPIHelper


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model Diagnostics", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")
    parser.add_argument("--model-name", type=str, default="cttsai", help="")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    _args = parse_commandline()

    root_resource_path: str = ft.root_resource_path
    configs = SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=_args.configs)

    #
    helper = NumerAPIHelper(root_dir_path=root_resource_path)
    helper.do_submission(model_name=_args.model_name, dir_path=configs.output_dir_, )
