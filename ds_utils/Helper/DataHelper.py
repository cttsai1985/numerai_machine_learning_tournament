import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple
from ds_utils import Utils
from ds_utils import DiagnosticUtils
from ds_utils import Metrics


class DataHelper:
    def __init__(
            self, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
            cols_group: Optional[List[str]] = None, on_disk: bool = True):
        self.filename: str = filename
        self.dataset_type: str = dataset_type
        self.col_target: str = col_target
        self.cols_feature: List[str] = cols_feature
        self.cols_group: Optional[List[str]] = cols_group

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
            raise FileNotFoundError(f"file {self.filename} does not exist.")

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
    def y_(self) -> pd.Series:
        data = self._read_data(columns=[self.col_target])
        return data.squeeze().astype("float")

    @property
    def y_name_(self) -> str:
        return self.col_target

    @property
    def X_(self) -> pd.DataFrame:
        return self._read_data(columns=self.cols_feature)

    @property
    def data_(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        return self.X_, self.y_, self.groups_


class EvaluationDataHelper(DataHelper):
    def __init__(
            self, filename: str, dataset_type: str, cols_feature: List[str], col_target: str = "target",
            cols_group: Optional[List[str]] = None, on_disk: bool = True):
        super().__init__(
            filename=filename, dataset_type=dataset_type, cols_feature=cols_feature, col_target=col_target,
            cols_group=cols_group, on_disk=on_disk)

    def pct_rank_normalize(self, yhat: pd.Series, ) -> pd.Series:
        logging.info(f"rank predictions for regression modeling")
        groups = self.groups_
        predictions = pd.concat([yhat, groups], axis=1)
        return predictions.groupby(groups.name)[yhat.name].apply(lambda x: Utils.pct_ranked(x))

    def feature_neutralize(
            self, yhat: pd.Series, col_yhat: str = "yhat", col_neutral: str = "neutral_yhat",
            cols_feature: Optional[List[str]] = None, proportion: float = 1., normalize: bool = True) -> pd.Series:
        if cols_feature is None:
            cols_feature = self.cols_feature

        groups = self.groups_
        groups_name = groups.name

        data = pd.concat([self.X_, yhat.rename(col_yhat), groups], axis=1)
        predictions = yhat.to_frame()
        predictions[col_neutral] = data.groupby(groups_name).apply(
            lambda x: DiagnosticUtils.compute_neutralize(
                x, target_columns=[col_yhat], neutralizers=cols_feature, proportion=proportion,
                normalize=normalize))
        return predictions[col_neutral].rename(col_yhat)

    def evaluate_with_feature(
            self, yhat: pd.Series, method: Optional[Callable] = None, cols_feature: Optional[List[str]] = None,
            **kwargs) -> pd.DataFrame:

        groups = self.groups_
        data = self.X_
        if not cols_feature:
            cols_feature = self.cols_feature

        if not method:
            method = Metrics.pearson_corr

        data: pd.DataFrame = pd.concat([data, yhat, groups], axis=1)
        results = data.groupby(groups.name).apply(lambda x: DiagnosticUtils.feature_exposure(
            x, columns_feature=cols_feature, column_target=yhat.name, method=method))
        # results.columns.name = ""
        return results

    def evaluate_with_y(
            self, yhat: pd.Series, scoring_func: Callable, col_yhat: str = "yhat", scoring_type: Optional[str] = None,
            **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        groups = self.groups_
        groups_name = groups.name

        y = self.y_
        y_name = y.name
        logging.info(f"evaluate_with_y: {y.name} ({y.shape})")

        predictions: pd.DataFrame = pd.concat([y, yhat.rename(col_yhat), groups], axis=1)
        gpdf_predictions = predictions.groupby(groups_name)

        score_split: pd.DataFrame = gpdf_predictions.apply(
            lambda x: scoring_func(x[y_name], x[col_yhat], scoring_type=scoring_type))
        score_split = score_split.reindex(index=Utils.natural_sort(score_split.index.tolist()))

        # all score
        score_all: pd.DataFrame = scoring_func(
            y, predictions[col_yhat], scoring_type=scoring_type).to_frame("evaluationMean")
        score_all["evaluationSharpeByEra"] = DiagnosticUtils.sharpe_ratio(score_split)
        score_all["evaluationMeanByEra"] = score_split.mean()
        score_all["evaluationStdByEra"] = score_split.std()
        score_all["evaluationSmartSharpeByEra"] = DiagnosticUtils.smart_sharpe(score_split)
        score_all = score_all.T
        score_all.index.name = "metrics"

        logging.info(f"Performance on target: {y_name}:\n{score_all}\n\n{score_split}\n\n{score_split.describe()}")
        return predictions, score_all, score_split
