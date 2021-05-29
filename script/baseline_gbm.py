import sys
import os
import json
import argparse
import logging
import hashlib
from pathlib import Path
from glob import glob
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

# EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent, "numerai_utils")
EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils
from ds_utils import Solution, SolutionConfigs, OptunaLGBMTuner
from ds_utils import RefreshLevel


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solver", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    root_resource_path: str = "../input/numerai_tournament_resource/"

    configs = SolutionConfigs(root_resource_path=root_resource_path, eval_data_type=args.eval_data_type,)

    output_data_path: str = configs.output_dir_
    Path(output_data_path).mkdir(mode=0o777, parents=True, exist_ok=True)
    if args.debug:
        configs.num_boost_round = 5

    tuner = OptunaLGBMTuner.from_configs(
        args=args, configs=configs, output_data_path=output_data_path).load_optimized_params()
    if args.compute_hpo:
        tuner.run(data_type=args.eval_data_type)

    configs.model_best_params = tuner.params_
    solver = Solution.from_configs(args=args, configs=configs, output_data_path=output_data_path)
    if args.compute_eval:
        solver.evaluate(train_data_type=args.eval_data_type, valid_data_type="validation")

    if args.compute_infer:
        solver.run()

    return


if "__main__" == __name__:
    ds_utils.initialize_logger()
    _args = parse_commandline()
    main(_args)
    # TODO: building
    # TODO: model helper
    # TODO: parameterize script, half way
    # TODO: Post processing
    # TODO: choose subset training set
    # TODO: fine tune model
    # TODO: add select era for model analytics

    # TODO: experimenting
    # TODO: multi class approach
