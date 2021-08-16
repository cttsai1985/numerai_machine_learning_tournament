import sys
import os
import json
import time
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

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
    logging.info(f'run command: {" ".join(command)}')

    # noinspection PyBroadException
    try:
        process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, )
        live_output(process)
        process.wait()
        time.sleep(1)

    except Exception:
        logging.info(f'failed running command: {" ".join(command)}')

    return


def compile_infer(script_type: str, script_file: str, config_file: str, refresh_level: str, command: List[str]):
    return [script_type, script_file, "--refresh-level", refresh_level, "--configs", config_file] + command


def compile_offline_diagnostics(
        script_type: str, script_file: str, config_file: str, command: Optional[List[str]] = None):
    ret = [script_type, script_file, "--configs", config_file]
    if command:
        ret += command
    return ret


def compile_online_diagnostics(
        script_type: str, script_file: str, config_file: str, model_name: str, numerai_public_id: str,
        numerai_secret: str, command: Optional[List[str]] = None):
    exec_command = [
        script_type, script_file, "--configs", config_file, "--model-name", model_name, "--numerapiPublicID",
        "${" + f"{numerai_public_id}" + "}", "--numerapiSecret", "${" + f"{numerai_secret}" + "}"]
    if command:
        exec_command += command
    return exec_command


def compile_base_command(
        script_type: str, script_file: str, **kwargs):
    exec_command = [script_type, script_file, ]
    return exec_command


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
    for execute_params in task_sequence:
        execute_config_file = execute_params["config_file"]
        logging.info(f"inference with: {execute_config_file}")
        execute_on_process(compile_infer(**execute_params))

        offline_diagnostic_configs = config_file["offline_diagnostics"]
        if offline_diagnostic_configs:
            logging.info(f"offline evaluate with: {execute_config_file}")
            for diagnostic_params in offline_diagnostic_configs:
                execute_on_process(compile_offline_diagnostics(**diagnostic_params, config_file=execute_config_file))

        logging.info(f"online evaluate with: {execute_config_file}")
        online_diagnostic_configs = config_file["online_diagnostics"]
        if online_diagnostic_configs:
            for diagnostic_params in online_diagnostic_configs:
                diagnostic_params.update(config_file["numerapi_configs"])
                execute_on_process(
                    compile_online_diagnostics(config_file=execute_config_file, **diagnostic_params))

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
    args = parser.parse_args()
    return args


if "__main__" == __name__:
    ds_utils.initialize_logger()
    ds_utils.configure_pandas_display()
    _args = parse_commandline()
    main(_args)
