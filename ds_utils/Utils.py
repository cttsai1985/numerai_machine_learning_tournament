import hashlib
import logging
import os
import yaml
import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Callable, Any, Dict, List, Tuple, Union


def get_file_hash(file_path: str, length: int = 8) -> str:
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()[:length]


def scale_uniform(series: pd.Series, middle: float = 0.5) -> pd.Series:
    return (series.rank(pct=True, method="first") - middle) / series.shape[0]


def auto_corr_penalty(x: pd.Series, lag: int = 1) -> float:
    n = x.shape[0]
    p = np.abs(x.autocorr(lag=lag))
    return np.sqrt(1 + 2 * np.sum([((n - i) / n) * p ** i for i in range(1, n)]))


# to neutralize any series by any other series
def neutralize_series(series: pd.Series, by: pd.Series, proportion: float = 1.0) -> pd.Series:
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack((exposures, np.array([series.mean()] * len(exposures)).reshape(-1, 1)))
    correction = proportion * (exposures.dot(np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


def atoi(text: str) -> Union[str, bool]:
    return int(text) if text.isdigit() else text


def natural_keys(text: str) -> List[str]:
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return list(map(atoi, re.split(r'(\d+)', text)))


def natural_sort(list_of_str: List[str]) -> List[str]:
    list_of_str.sort(key=natural_keys)
    return list_of_str


def load_yaml_configs(configs_file_path: str) -> Dict[str, Any]:
    if not all([os.path.exists(configs_file_path), os.path.isfile(configs_file_path)]):
        logging.info(f"configs does not exist: {configs_file_path}")
        return dict()

    with open(configs_file_path, 'r') as fp:
        ret = yaml.load(fp, Loader=yaml.FullLoader)

    return ret


def save_yaml_configs(configs: Dict[str, Any], configs_file_path: str, debug: bool = False) -> Dict[str, Any]:
    dir_path: str = Path(configs_file_path).parent
    if not all([os.path.exists(dir_path), os.path.isdir(dir_path)]):
        logging.info(f"create directory: {dir_path}")

    Path(dir_path).mkdir(parents=True, exist_ok=True)
    with open(configs_file_path, 'w') as fp:
        yaml.dump(configs, fp)
        logging.info(f"write to: {configs_file_path}, with: {configs}")

    if debug:
        configs = load_yaml_configs(configs_file_path)

    return configs


if "__main__" == __name__:
    alist = ["something1", "something12", "something17", "something2", "something25", "something29"]
    print(alist, natural_sort(alist))
