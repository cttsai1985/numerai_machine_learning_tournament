import sys
import os
import json
import time
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List

EXTERNAL_UTILS_LIB = os.path.join(Path().resolve().parent)
sys.path.insert(0, EXTERNAL_UTILS_LIB)

import ds_utils


def parse_commandline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="execute a series of scripts", add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs", type=str, default="infer_configs.yaml", help="configs file")
    args = parser.parse_args()
    return args


def live_output(process):
    # Poll process.stdout to show stdout live
    while True:
        output = process.stdout.readline()
        if not process.poll():
            break

        if output:
            print(output.strip())
            sys.stdout.flush()

    return process.poll()


def execute_on_process(command: List[str]):
    logging.info(f'run command: {" ".join(command)}')
    process = subprocess.Popen(" ".join(command), shell=True, stdout=subprocess.PIPE, )
    live_output(process)
    time.sleep(1)
    return


def compile_infer(script_type: str, script_file: str, config_file: str, refresh_level: str, command: List[str]):
    return [script_type, script_file, "--refresh-level", refresh_level, "--configs", config_file] + command


def compile_online_diagnostics(
        script_type: str, script_file: str, configs_file: str, model_name: str, numerai_public_id: str,
        numerai_secret: str):
    exec_command = [
        script_type, script_file, "--configs", configs_file, "--model-name", model_name, "--numerapiPublicID",
        "${" + f"{numerai_public_id}" + "}", "--numerapiSecret", "${" + f"{numerai_secret}" + "}"]
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

    for execute_params in config_file.get("preprocess", list()):
        execute_on_process(compile_base_command(**execute_params))

    for execute_params in config_file.get("compute", list()):
        execute_config_file = execute_params["config_file"]
        logging.info(f"inference with: {execute_config_file}")
        execute_on_process(compile_infer(**execute_params))
        logging.info(f"online evaluate with: {execute_config_file}")
        execute_on_process(compile_online_diagnostics(configs_file=execute_config_file, **config_file["diagnostics"]))

    for execute_params in config_file.get("postprocess", list()):
        execute_on_process(compile_base_command(**execute_params))

    return


if "__main__" == __name__:
    ds_utils.initialize_logger()
    _args = parse_commandline()
    main(_args)
