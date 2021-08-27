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
from ds_utils import Solution, SolutionConfigs, OptunaLGBMTuner
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
    root_resource_path: str = "../input/numerai_tournament_resource/"

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
    # TODO: P0
    # TODO: Modeling
    # cross-entropy application: xentropy (cross_entropy), xentlambda (cross_entropy_lambda)
    # to replace era boosting by using GOSS or force to run low sampling ratio
    # choose subset of training set for training
    # fine tune model using recent data, incremental learning
    # adding more parameters to tune
    # sklearn stacking: StackingClassifier

    # TODO: P0
    # Feature Engineering
    # stats features on row-levels: skew, kurt, mean, std, frequency
    # sparse transform: tree-embedding for deep features interactions
    # dimensionality reduction by change of basis: PCA, Kernel-PCA, tSVD, UMAP
    # dimensionality reduction by clustering, distance matrix: correlations, normalized mutual information

    # TODO: P0
    # TODO: Monitoring
    # compute and save feature importance stats

    # TODO: P1
    # TODO: Experimenting
    # multi class approach
    # ranker approach
    # compute AUC-ROC on multi-labels classification (macro or micro)
    # dynamic parameters such as learning_rate using call back
    # categorical_feature in LightGBM, unordered categorical

    # TODO: P2
    # TODO: Monitoring
    # investigate mmc issue
    # add select era for model analytics

    # TODO: P2
    # TODO: Utilities
    # persistent save/load yaml

    # TODO: long-term
    # stationary features (specific goals needed here)
    # parameterize script (specific goals needed here)
    # model helper (specific goals needed here)
