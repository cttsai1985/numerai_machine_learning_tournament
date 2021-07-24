import sys
import os
import json
import logging
import numpy as np
import dask.array as da
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from ds_utils.Utils import scale_uniform, natural_sort


class DataHelper:
    def __init__(
            self, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
            cols_group: Optional[List[str]] = None):
        self.filename: str = filename
        self.dataset_type: str = dataset_type
        self.col_target: str = col_target
        self.cols_feature: List[str] = cols_feature
        self.cols_group: Optional[List[str]] = cols_group
        self.cols_group_for_eval: List[str] = ["era"]

        self.data: Optional[pd.DataFrame] = None
        self.is_loaded: bool = False
        self.is_good: bool = False

    @classmethod
    def from_params(cls, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
                    cols_group: Optional[List[str]] = None):
        return cls(
            filename=filename, dataset_type=dataset_type, cols_feature=cols_feature, col_target=col_target,
            cols_group=cols_group)

    @classmethod
    def from_configs(cls, **kwargs):
        # TODO: implemented and refactor other dependency
        return NotImplementedError()

    def load(self):
        if not os.path.exists(self.filename):
            logging.warning(f"file {self.filename} does not exist.")
            return self

        self.data = pd.read_parquet(self.filename).set_index("id")
        logging.info(f"read {self.data.shape} from {self.filename}")
        self.is_loaded = True

        if self.col_target not in self.data.columns:
            logging.warning(f"target {self.col_target} does not exist in data.columns.")

        _cols = set(self.data.columns.tolist())
        if self.cols_group and not set(self.cols_group).issubset(_cols):
            logging.warning(f"groups {self.cols_group} does not exist in data.columns.")

        if not set(self.cols_feature).issubset(_cols):
            _diff = _cols - set(self.cols_feature)
            logging.warning(f"not all feature columns: missing {len(_diff)}: {_diff}")

        self.is_good = True
        return self

    def process(self, process_funcs: Optional[List[Callable]] = None):
        if not all([process_funcs, self.is_good, self.is_loaded]):
            return self

        data, y, groups = self.data_
        self.data = pd.concat([data] + [func(data, y, groups) for func in process_funcs], axis=1)
        # TODO: feature transform
        return self

    def feature_neutralize(self, yhat: pd.Series) -> pd.Series:
        # Normalize submission
        scale_uniform_yhat = scale_uniform(yhat).values

        # Neutralize submission to features
        f = self.X_.values
        neutralized_x = da.asarray(f).dot(da.asarray(np.linalg.pinv(f)).dot(scale_uniform_yhat)).compute()

        neutralized_yhat = scale_uniform_yhat - neutralized_x
        ret = self.groups_.to_frame()
        ret["yhat"] = neutralized_yhat
        ret["std"] = ret.groupby(self.group_name_).transform("std")
        ret["yhat"] /= ret["std"]
        return ret.reindex(columns=["yhat"]).squeeze()

    def evaluate(
            self, yhat: pd.Series, scoring_func: Callable, col_yhat: str = "yhat",
            feature_neutral: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        scoring_type = None
        if feature_neutral:
            logging.info(f"evaluate with neutralizing prediction by feature")
            yhat = self.feature_neutralize(yhat)
            scoring_type = "corr"

        predictions = pd.concat([self.y_, yhat.rename(col_yhat), self.groups_for_eval_], axis=1)
        score_split = predictions.groupby(self.group_name_for_eval_).apply(
            lambda x: scoring_func(x[self.y_.name], x[col_yhat], scoring_type=scoring_type))
        score_split = score_split.reindex(index=natural_sort(score_split.index.tolist()))
        score_all = scoring_func(
            self.y_, predictions[col_yhat], scoring_type=scoring_type).to_frame(self.group_name_for_eval_)
        score_all["sharpe"] = score_split.mean() / score_split.std()
        score_all = score_all.T
        logging.info(f"Performance:\n{score_all}\n\n{score_split}\n\n{score_split.describe()}")
        return predictions, score_all, score_split

    def save(self, filepath: Optional[str] = None):
        if not self.is_loaded:
            logging.warning(f"not to save since data is not good")
            return self

        if not filepath:
            filepath = self.filename

        # TODO: create folder before saving?
        self.data.reset_index().to_parquet(filepath)
        logging.info(f"save {self.dataset_type} data into {filepath}")
        return self

    @property
    def group_name_(self) -> str:
        return "_".join(self.cols_group)

    @property
    def groups_(self) -> pd.Series:
        if not self.cols_group:
            return pd.Series()

        if len(self.cols_group) == 1:
            return self.data.reindex(columns=self.cols_group).squeeze()

        return self.data.reindex(
            columns=self.cols_group).apply(lambda x: "_".join(x.astype("str")), axis=1).rename(self.group_name_)

    @property
    def groups_for_eval_(self) -> pd.Series:
        return self.data.reindex(columns=self.cols_group_for_eval).squeeze()

    @property
    def group_name_for_eval_(self) -> str:
        return "_".join(self.cols_group_for_eval)

    @property
    def y_(self) -> pd.Series:
        return self.data[self.col_target].astype("float")

    @property
    def X_(self) -> pd.DataFrame:
        return self.data.reindex(columns=self.cols_feature)

    @property
    def data_(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        return self.X_, self.y_, self.groups_
