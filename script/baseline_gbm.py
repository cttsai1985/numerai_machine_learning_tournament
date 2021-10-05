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
from ds_utils import FilenameTemplate as ft
from ds_utils import AutoSolution, SolutionConfigs, OptunaLGBMTuner
from ds_utils import RefreshLevel

ds_utils.configure_pandas_display()


def mkdir_safe(dir_path: str, desired_permission=0o755):
    try:
        original_umask = os.umask(0)
        Path(dir_path).mkdir(mode=desired_permission, parents=True, exist_ok=True)
    finally:
        os.umask(original_umask)


def chmod_safe(dir_path: str, desired_permission=0o766):
    try:
        original_umask = os.umask(0)
        tmp = list(glob(os.path.join(dir_path, "*")))
        logging.info(f"chmod to {tmp}")
        map(lambda x: os.chmod(x, mode=desired_permission), tmp)
    finally:
        os.umask(original_umask)


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solver", add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="./configs.yaml", help="configs file")
    parser.add_argument(
        "--eval-data-type", type=str, default="training", help="data for training and evaluation on cross validation")

    parser.add_argument(
        "--refresh-level", type=str, default=None,
        help=f"determine the level of computing: {RefreshLevel.possible_refresh_levels()}")
    parser.add_argument("--compute-hpo", action="store_true", help="run hyper-parameter tuning")
    parser.add_argument("--compute-eval", action="store_true", help="run cross validation and validation")
    parser.add_argument("--compute-infer", action="store_true", help="run inference")
    parser.add_argument("--compute-all", action="store_true", help="run all compute")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    if args.debug:
        args.eval_data_type = "validation"

    args.refresh_level = RefreshLevel(args.refresh_level)
    if args.compute_all:
        args.compute_hpo = True
        args.compute_eval = True
        args.compute_infer = True
    return args


def main(args: argparse.Namespace):
    root_resource_path: str = ft.root_resource_path

    configs = SolutionConfigs(
        root_resource_path=root_resource_path, configs_file_path=args.configs, eval_data_type=args.eval_data_type, )

    output_data_path: str = configs.output_dir_
    Path(output_data_path).mkdir(parents=True, exist_ok=True)
    if args.debug:
        configs.num_boost_round = 5

    tuner = OptunaLGBMTuner.from_configs(args=args, configs=configs, output_data_path=output_data_path)
    if args.compute_hpo:
        tuner.run(data_type=args.eval_data_type)

    configs.load_optimized_params_from_tuner(tuner=tuner)
    solver = AutoSolution.from_configs(args=args, configs=configs, output_data_path=output_data_path)
    if args.compute_eval:
        solver.evaluate(train_data_type=args.eval_data_type, valid_data_type="validation")

    if args.compute_infer:
        solver.run()

    return


if "__main__" == __name__:
    ds_utils.initialize_logger()
    _args = parse_commandline()
    main(_args)

