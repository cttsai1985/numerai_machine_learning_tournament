import hashlib
import logging
import os
import yaml
import re
import json
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union
import pandas as pd


def scale_uniform(series: pd.Series, middle: float = 0.5) -> pd.Series:
    return (series.rank(pct=False, method="first") - middle) / series.shape[0]


def pct_ranked(series: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    return series.rank(method="dense", ascending=True, pct=True)


def select_series_from_tb_num(series: pd.Series, groups: pd.Series, tb_num: int) -> pd.Series:
    data = pd.concat([series, groups], axis=1, sort=False)
    series_name = series.name
    samples = data.groupby(
        groups)[series_name].apply(lambda x: pd.concat([x.nlargest(n=tb_num), x.nsmallest(n=tb_num)]))
    return samples.reset_index(groups.name)[series_name]


def get_file_hash(file_path: str, length: int = 8) -> str:
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()[:length]


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
