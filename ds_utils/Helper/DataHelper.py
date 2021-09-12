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
from ds_utils.Utils import scale_uniform, pct_ranked, natural_sort
from ds_utils.DiagnosticUtils import sharpe_ratio


class DataHelper:
    def __init__(
            self, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
            cols_group: Optional[List[str]] = None, on_disk: bool = True):
        self.filename: str = filename
        self.dataset_type: str = dataset_type
        self.col_target: str = col_target
        self.cols_feature: List[str] = cols_feature
        self.cols_group: Optional[List[str]] = cols_group
        self.cols_group_for_eval: List[str] = ["era"]

        self.on_disk: bool = on_disk

        self.data: Optional[pd.DataFrame] = None
        self.is_loaded: bool = False
        self.is_good: bool = False

    @classmethod
    def from_params(cls, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
                    cols_group: Optional[List[str]] = None, on_disk: bool = True):
        return cls(
            filename=filename, dataset_type=dataset_type, cols_feature=cols_feature, col_target=col_target,
            cols_group=cols_group, on_disk=on_disk)

    @classmethod
    def from_configs(cls, **kwargs):
        # TODO: implemented and refactor other dependency
        return NotImplementedError()

    def _read_data(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if self.on_disk:
            data = pd.read_parquet(self.filename, columns=columns)
        else:
            data = self.data.reindex(columns=columns)

        if data.index.name != "id":
            data.set_index("id", inplace=True)

        return data

    def load(self, reload: bool = False):
        if not os.path.exists(self.filename):
            logging.warning(f"file {self.filename} does not exist.")
            raise ValueError(f"file {self.filename} does not exist.")

        if self.on_disk:
            logging.info(f"run on_disk to save memory")
            return self

        elif not self.on_disk and self.is_loaded and not reload:
            logging.info(f"file {self.filename} is is_loaded and not reloaded.")
            return self

        self.data = self._read_data()
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

    def pct_rank_normalize(self, yhat: pd.Series, ) -> pd.Series:
        logging.info(f"rank predictions for regression modeling")
        predictions = pd.concat([yhat, self.groups_for_eval_], axis=1)
        return predictions.groupby(self.group_name_for_eval_)[yhat.name].apply(lambda x: pct_ranked(x))

    def feature_neutralize(self, yhat: pd.Series) -> pd.Series:
        # Normalize submission
        scale_uniform_yhat = scale_uniform(yhat).values

        # Neutralize submission to features
        f = self.X_.values
        neutralized_x = da.asarray(f).dot(da.asarray(np.linalg.pinv(f)).dot(scale_uniform_yhat)).compute()

        neutralized_yhat = scale_uniform_yhat - neutralized_x
        ret = self.groups_for_eval_.to_frame()
        ret["yhat"] = neutralized_yhat
        ret["yhat_std"] = ret.groupby(self.group_name_for_eval_).transform("std")
        ret["yhat"] /= ret["yhat_std"]
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

        # all score
        score_all = scoring_func(
            self.y_, predictions[col_yhat], scoring_type=scoring_type).to_frame(self.group_name_for_eval_)
        score_all["sharpe"] = sharpe_ratio(score_split)
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

        data = self._read_data(columns=self.cols_group)

        if len(self.cols_group) == 1:
            return data.squeeze().astype("category")

        return data.apply(lambda x: "_".join(x.astype("str")), axis=1).astype("category").rename(self.group_name_)

    @property
    def groups_for_eval_(self) -> pd.Series:
        return self._read_data(columns=self.cols_group_for_eval).squeeze()

    @property
    def group_name_for_eval_(self) -> str:
        return "_".join(self.cols_group_for_eval)

    @property
    def y_(self) -> pd.Series:
        data = self._read_data(columns=[self.col_target])
        return data.squeeze().astype("float")

    @property
    def X_(self) -> pd.DataFrame:
        return self._read_data(columns=self.cols_feature)

    @property
    def data_(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        return self.X_, self.y_, self.groups_
