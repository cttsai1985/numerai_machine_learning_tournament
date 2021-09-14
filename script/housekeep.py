import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import pdb
from glob import glob
from typing import Tuple, Dict, List
from dask import dataframe as dd
import shutil

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs


def parse_commandline() -> argparse.Namespace:
    default_destination_output: str = "../input/numerai_tournament_resource/old/"
    default_destination_configs: str = "../configs/not_active/"
    default_configs_pattern: str = "../configs/configs_baseline_*.yaml"
    default_output_pattern: str = "lgbm*"
    default_output_file: str = "model.pkl"
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="dry run")
    parser.add_argument(
        "--destination-output", type=str, default=default_destination_output, help="destination to move output dirs")
    parser.add_argument(
        "--destination-configs", type=str, default=default_destination_configs, help="destination to move configs")
    parser.add_argument(
        "--configs-pattern", type=str, default=default_configs_pattern, help="default configs patten")
    parser.add_argument(
        "--output-pattern", type=str, default=default_output_pattern, help="default output patten")
    parser.add_argument(
        "--output-file-lookup", type=str, default=default_output_file, help="default output file to lookup")
    parser.add_argument("--num-rows", type=int, default=100, help="display rows per attributes")

    args = parser.parse_args()
    return args


def process_file_path(file_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    logging.info(f"process pattern as: {file_path}")
    original_path: List[str] = list(map(lambda x: str(Path(x).parent.absolute()), glob(file_path)))
    path_df = pd.DataFrame({"original_path": original_path})
    path_df["path"] = path_df["original_path"].apply(lambda x: Path(x).parts[-1])
    path_df['configs_path'] = path_df["original_path"].map(config_file_output_dict)
    path_df['configs_available'] = path_df["original_path"].isin(active_output_dirs)
    output_dirs_to_move = path_df.loc[~path_df['configs_available'], 'original_path'].tolist()
    output_dirs_to_keep = path_df.loc[path_df['configs_available'], 'original_path'].tolist()
    return output_dirs_to_move, output_dirs_to_keep


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    col_metric: str = "attr"
    root_resource_path: str = "../input/numerai_tournament_resource/"

    _args = parse_commandline()
    Path(_args.destination_output).mkdir(parents=True, exist_ok=True)
    Path(_args.destination_configs).mkdir(parents=True, exist_ok=True)

    existing_config_files = list(glob(_args.configs_pattern))
    logging.info(f"find {len(existing_config_files)} configs: {existing_config_files}")

    logging.info(f"start loading configs:")
    existing_configs = list(map(
        lambda x: SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=x), existing_config_files))

    output_dirs = list(map(lambda x: str(Path(x.output_dir_).absolute()), existing_configs))
    config_file_output_dict = {k: v for k, v in zip(output_dirs, existing_config_files)}

    active_output_dirs = list(filter(lambda x: os.path.exists(x) and os.path.isdir(x), output_dirs))
    non_active_output_dirs = list(filter(lambda x: ~os.path.exists(x) or ~os.path.isdir(x), output_dirs))

    file_pattern: str = os.path.join(
        root_resource_path, _args.output_pattern, _args.output_file_lookup)

    _output_dirs_to_move, _output_dirs_to_keep = process_file_path(file_pattern)
    logging.info(f"move {len(_output_dirs_to_move)} output directories to {_args.destination_output}")
    for dir_to_move in _output_dirs_to_move:
        if _args.dry_run:
            logging.info(f"DRY RUN: moving dir {dir_to_move} to {_args.destination_output}")
            continue

        # noinspection PyBroadException
        try:
            shutil.move(dir_to_move, _args.destination_output)
            logging.info(f"moving dir {dir_to_move} to {_args.destination_output}")

        except Exception:
            logging.error(f"failed to moving dir {dir_to_move} to {_args.destination_output}")

    config_files_to_move = [v for k, v in config_file_output_dict.items() if k not in _output_dirs_to_keep]
    logging.info(f"move {len(config_files_to_move)} configs file to {_args.destination_configs}")
    for config_to_move in config_files_to_move:
        if _args.dry_run:
            logging.info(f"DRY RUN: moving dir {config_to_move} to {_args.destination_configs}")
            continue

        # noinspection PyBroadException
        try:
            shutil.move(config_to_move, _args.destination_configs)
            logging.info(f"moving dir {config_to_move} to {_args.destination_configs}")

        except Exception:
            logging.error(f"failed to moving file {config_to_move} to {_args.destination_configs}")
