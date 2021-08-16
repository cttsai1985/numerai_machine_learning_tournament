import sys
import os
import argparse
import logging
from glob import glob
from typing import List
from itertools import combinations_with_replacement
import pdb
import pandas as pd
from dask import dataframe as dd
from pathlib import Path
from typing import Callable

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import SolutionConfigs


def parse_commandline() -> argparse.Namespace:
    ensemble_template_file_path: str = "./template_ensemble_configs.yaml"
    run_template_file_path: str = "./template_empty_ensemble_run_configs.yaml"

    default_output_dir: str = "../ensemble_configs/"
    default_run_configs_filepath: str = "../infer_configs/batch_ensemble_infer.yaml"

    default_configs_pattern: str = "../configs/configs_baseline_*.yaml"
    default_model_name_stem: str = "ensemble_sel"

    parser = argparse.ArgumentParser(
        description="create a set of ensemble configs to try", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--run-configs-filepath", type=str, default=default_run_configs_filepath, help="output run configs file path")
    parser.add_argument(
        "--model-name-stem", type=str, default=default_model_name_stem, help="model name stem to use for output")
    parser.add_argument(
        "--ensemble-template", type=str, default=ensemble_template_file_path, help="ensemble template file path")
    parser.add_argument(
        "--run-template", type=str, default=run_template_file_path, help="run template file path")
    parser.add_argument(
        "--configs-pattern", type=str, default=default_configs_pattern, help="default configs patten")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="directory to write out")
    parser.add_argument("--low", type=int,default=2, help="min number of files to ensemble")
    parser.add_argument("--high", type=int, default=6, help="high number of files to ensemble")

    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()

    root_resource_path: str = "../input/numerai_tournament_resource/"

    _args = parse_commandline()
    _configs_template = ds_utils.Utils.load_yaml_configs(_args.ensemble_template)
    logging.info(f"read ensemble template: {_configs_template}")

    Path(_args.output_dir).mkdir(parents=True, exist_ok=True)

    existing_config_files = list(glob(_args.configs_pattern))
    logging.info(f"find {len(existing_config_files)} configs: {existing_config_files}")

    logging.info(f"start loading configs:")
    existing_configs = list(map(
        lambda x: SolutionConfigs(root_resource_path=root_resource_path, configs_file_path=x), existing_config_files))

    output_dirs = list(map(lambda x: str(Path(x.output_dir_).absolute()), existing_configs))

    active_output_dirs = list(filter(lambda x: os.path.exists(x) and os.path.isdir(x), output_dirs))
    config_file_output_dict = {k: v for k, v in zip(output_dirs, existing_config_files) if k in active_output_dirs}

    existing_config_files = list(config_file_output_dict.values())
    logging.info(f"available configs: {len(existing_config_files)}: {existing_config_files}")

    combinations_all_ranges = list()
    for i in range(_args.low, _args.high + 1):
        combinations = combinations_with_replacement(existing_config_files, i)
        combinations_all_ranges.extend(list(filter(lambda x: len(set(x)) > 1, combinations)))

    configs_all_ranges: List[str] = list()
    combinations_all_ranges = list(map(list, combinations_all_ranges))
    logging.info(f"available configs: {len(combinations_all_ranges)}")
    for i, pair in enumerate(combinations_all_ranges):
        _configs_template_copy = _configs_template.copy()

        model_name = "_".join([_args.model_name_stem, str(len(pair)), _configs_template_copy["ensemble_method"]])
        _configs_template_copy["model_name"] = model_name
        _configs_template_copy["ensemble_model_configs"] = pair

        ds_utils.Utils.save_yaml_configs(_configs_template_copy, "./tmp.yaml")
        # ds_utils.Utils.load_yaml_configs("./tmp.yaml")

        hash_str = ds_utils.Utils.get_file_hash("./tmp.yaml")
        filepath: str = os.path.join(_args.output_dir, f"ensemble_{i:04d}_{hash_str}.yaml")
        ds_utils.Utils.save_yaml_configs(_configs_template_copy, filepath)
        configs_all_ranges.append(filepath)

    # create run configs
    run_template = ds_utils.Utils.load_yaml_configs(_args.run_template)
    run_compute_collection = list()
    for config_filepath in configs_all_ranges:
        _run_compute_template = {
            "command": ["--compute-eval", ],
            "config_file": config_filepath,
            "refresh_level": "predictions",
            "script_file": "./ensemble.py",
            "script_type": "python"
        }
        # _run_compute_template["config_file"] = config_filepath
        run_compute_collection.append(_run_compute_template)

    run_template["compute"] = run_compute_collection
    ds_utils.Utils.save_yaml_configs(run_template, _args.run_configs_filepath)
