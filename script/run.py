import sys
import os
import json
import time
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Optional, Tuple

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils


def live_output(process):
    # Poll process.stdout to show stdout live
    while True:
        # noinspection PyBroadException
        try:
            output = process.stdout.readline()
            if not process.poll():
                break

            if output:
                print(output.strip())
                sys.stdout.flush()

        except Exception:
            logging.info(f'failed capture output')
            break

    return process.poll()


def execute_on_process(command: List[str]):
    # noinspection PyBroadException
    try:
        _command = " ".join(filter(lambda x: isinstance(x, str), command))
        logging.info(f"run command: {_command}")

    except Exception:
        logging.error(f"failed building command: {command}")

    # noinspection PyBroadException
    try:
        process = subprocess.Popen(_command, shell=True, stdout=subprocess.PIPE, )
        live_output(process)
        process.wait()
        time.sleep(1)

    except Exception:
        logging.error(f"failed running command: {_command}")

    return


def compile_infer(
        script_type: str, script_file: str, config_file: str, refresh_level: str, command: Optional[List[str]] = None):
    exec_command = [script_type, script_file, "--configs", config_file, "--refresh-level", refresh_level]
    if command is not None and command:
        exec_command += command
    return exec_command


def compile_offline_diagnostics(
        script_type: str, script_file: str, config_file: str, command: Optional[List[str]] = None):
    exec_command = [script_type, script_file, "--configs", config_file]
    if command is not None and command:
        exec_command += command
    return exec_command


def compile_online_diagnostics(
        script_type: str, script_file: str, config_file: str, model_name: str,
        command: Optional[List[str]] = None, **kwargs):
    exec_command = [script_type, script_file, "--configs", config_file, "--model-name", model_name, ]
    if command is not None and command:
        exec_command += command
    return exec_command


def compile_base_command(
        script_type: str, script_file: str, **kwargs):
    exec_command = [script_type, script_file, ]
    return exec_command


def compute_inference(params: Tuple):
    execute_params, config_file, i, num_task_sequence = params
    logging.info(f"inference: {i + 1:8d} / {num_task_sequence}")
    execute_config_file = execute_params["config_file"]
    logging.info(f"inference with: {execute_config_file}")
    execute_on_process(compile_infer(**execute_params))
    return


def compute_offline_evaluation(params: Tuple):
    execute_params, config_file, i, num_task_sequence = params
    logging.info(f"offline evaluation: {i + 1:8d} / {num_task_sequence}")

    execute_config_file = execute_params["config_file"]
    offline_diagnostic_configs = config_file["offline_diagnostics"]
    if not offline_diagnostic_configs:
        return

    logging.info(f"offline evaluate with: {execute_config_file}")
    for diagnostic_params in offline_diagnostic_configs:
        execute_on_process(compile_offline_diagnostics(**diagnostic_params, config_file=execute_config_file))

    return


def compute_online_evaluation(params: Tuple):
    execute_params, config_file, i, num_task_sequence = params
    logging.info(f"online evaluation: {i + 1:8d} / {num_task_sequence}")

    execute_config_file = execute_params["config_file"]
    logging.info(f"online evaluate with: {execute_config_file}")
    online_diagnostic_configs = config_file["online_diagnostics"]
    if not online_diagnostic_configs:
        return

    for diagnostic_params in online_diagnostic_configs:
        diagnostic_params.update(config_file["numerapi_configs"])
        execute_on_process(
            compile_online_diagnostics(config_file=execute_config_file, **diagnostic_params))

    return


def main(args: argparse.Namespace):
    if not (os.path.exists(_args.configs) and os.path.isfile(_args.configs)):
        logging.error(f"configuration is not usable. abort: {_args.configs}")
        return

    with open(args.configs, 'r') as fp:
        config_file = yaml.load(fp, Loader=yaml.FullLoader)

    task_sequence = config_file.get("preprocess")
    if not task_sequence:
        task_sequence = list()
    for execute_params in task_sequence:
        execute_on_process(compile_base_command(**execute_params))

    task_sequence = config_file.get("compute")
    if not task_sequence:
        task_sequence = list()

    _num_task_sequence = len(task_sequence)
    task_sequence = [
        (_execute_params, config_file, i, _num_task_sequence) for i, _execute_params in enumerate(task_sequence)]

    if args.n_jobs == 1:
        list(map(lambda x: compute_inference(x), task_sequence))
        list(map(lambda x: compute_offline_evaluation(x), task_sequence))
    else:
        n_jobs = args.n_jobs if args.n_jobs > 1 else cpu_count()
        logging.info(f"multi processing: {n_jobs}")
        with Pool(n_jobs) as p:
            p.map(compute_inference, task_sequence)

            offline_diagnostic_configs = config_file["offline_diagnostics"]
            if offline_diagnostic_configs:
                p.map(compute_offline_evaluation, task_sequence)
            else:
                logging.info(f"skip offline evaluation")

    online_diagnostic_configs = config_file["online_diagnostics"]
    if online_diagnostic_configs:
        list(map(compute_online_evaluation, task_sequence))
    else:
        logging.info(f"skip online evaluation")

    task_sequence = config_file.get("closure")
    if not task_sequence:
        task_sequence = list()
    for execute_params in task_sequence:
        execute_on_process(compile_base_command(**execute_params))

    return


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="infer_configs.yaml", help="configs file")
    parser.add_argument("--n-jobs", type=int, default=1, help="how many process to use")
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()
    _args = parse_commandline()
    main(_args)
