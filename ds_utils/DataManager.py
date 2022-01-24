import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple, Union

from ds_utils.Helper import DataHelper, EvaluationDataHelper, NeutralizationDataHelper
from ds_utils import FilenameTemplate

_DATA_PAIRS: List[Tuple[str]] = FilenameTemplate.numerai_data_filename_pairs
_EXAMPLE_PAIRS: List[Tuple[str]] = FilenameTemplate.numerai_example_filename_pairs


class DataManager:
    def __init__(
            self, working_dir: str = "./", data_mapping: Dict[str, str] = None, col_target: str = "target",
            cols_group: Optional[List[str]] = None, cols_feature: Optional[List[str]] = None,
            col_neutralization_reference: str = "target_nomi_60", neutralization_proportion: float = 0.5, **kwargs):
        self.working_dir: str = working_dir if working_dir else "./"

        self.example_mapping: Dict[str, str] = dict(_EXAMPLE_PAIRS)
        self.data_file_types: List[str] = list(zip(*_DATA_PAIRS))[0]
        self.data_file_names: List[str] = list(zip(*_DATA_PAIRS))[1]
        self.data_mapping: Dict[str, str] = dict(_DATA_PAIRS)

        if data_mapping:
            self.data_mapping = data_mapping
            self.data_file_types = list(data_mapping.keys())
            self.data_file_names = list(data_mapping.values())

        self._check_status()
        self.col_target: str = col_target
        self.cols_group: Optional[List[str]] = cols_group if cols_group and isinstance(cols_group, list) else ["era"]
        self.cols_feature: Optional[List[str]] = cols_feature

        self.col_neutralization_reference: Optional[str] = col_neutralization_reference
        self.neutralization_proportion: float = neutralization_proportion

        self.label2index: Optional[Dict[int, float]] = None
        self.index2label: Optional[Dict[float, int]] = None

    def _create_label_index_mapping(self) -> Dict[int, float]:
        sample_data = self.get_data_helper_by_type(data_type="training", for_evaluation=False)
        y_ = sample_data.y_.dropna().unique()
        y_.sort()
        return {v: i for i, v in enumerate(y_)}

    def configure_label_index_mapping(self):
        self.label2index = self._create_label_index_mapping()
        self.index2label = {v: k for k, v in self.label2index.items()}
        return self

    @classmethod
    def from_configs(cls, configs: "SolutionConfigs", **kwargs):
        # TODO: implemented and refactor other dependency

        column_neutralization_reference: Optional[str] = None
        neutralization_proportion: float = 0.5
        if configs.column_neutralization_reference is not None:
            column_neutralization_reference = configs.column_neutralization_reference
            neutralization_proportion = configs.neutralization_proportion

        # configs.load_feature_columns_from_json()
        return cls(
            working_dir=configs.input_data_dir, data_mapping=configs.data_mapping, col_target=configs.column_target,
            cols_group=configs.columns_group, cols_feature=configs.columns_feature,
            cols_neutralization_reference=column_neutralization_reference,
            neutralization_proportion=neutralization_proportion)

    def _check_status(self):
        logging.info(f"found {len(self.data_mapping)} data: in {self.data_mapping.keys()}")
        for k, v in self.data_mapping.items():
            filepath = os.path.join(self.working_dir, v)
            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                logging.warning(f"mapped file: {v} (data_type: {k}) not in working directory: {self.working_dir}")
                continue

        return self

    def has_cast_mapping(self, cast_type: Optional[str] = None) -> bool:
        if cast_type not in ["label2index", "index2label"]:
            return False

        cast_func: Optional[Callable] = getattr(self, cast_type, None)
        if cast_func is None or not cast_func:
            return False

        return True

    @staticmethod
    def impute_target(y: pd.Series, na_value: Optional[float] = None) -> pd.Series:
        if y.isna().any():
            logging.info(f"labels contained nan: {sorted(y.unique().tolist())}")
            if na_value is None:
                na_value = 0.5

            y.fillna(na_value, inplace=True)

        return y

    def cast_target(self, y: pd.Series, cast_type: Optional[str] = None) -> pd.Series:
        if cast_type is None:
            return y

        index_mapping = getattr(self, cast_type, None)
        if not index_mapping:
            return y

        y = self.impute_target(y, na_value=0.5)
        _y = y.map(index_mapping)
        _y_isna = _y.isna()
        if _y_isna.any():
            logging.info(f"mapping is not completed: {_y_isna.sum()} with {y.unique()}")
            raise ValueError(f"mapping is not completed: {_y_isna.sum()} with {y.unique()}")

        return _y

    def cast_target_proba(self, y: pd.Series, cast_type: Optional[str] = None) -> pd.Series:
        if cast_type is None:
            return y

        index_mapping = getattr(self, cast_type, None)
        if not index_mapping:
            return y

        yhat = pd.Series(np.dot(y, list(self.index2label.values)), index=y.index)
        yhat.name = y.name
        return yhat

    def get_example_data_by_type(self, data_type: str = "tournament") -> pd.DataFrame:
        if data_type not in self.example_mapping:
            return pd.DataFrame()

        filepath: str = os.path.join(self.working_dir, self.example_mapping.get(data_type))
        if not (os.path.exists(filepath) and os.path.isfile(filepath)):
            return pd.DataFrame()

        return pd.read_parquet(filepath)

    def get_data_helper_by_type(
            self, data_type: str = "training", preload: bool = True,
            for_evaluation: bool = False) -> Union[DataHelper, EvaluationDataHelper]:
        if data_type not in self.data_mapping.keys():
            raise ValueError(f"{data_type} not in data mapping")

        filename = os.path.join(self.working_dir, self.data_mapping.get(data_type))
        cols_feature = self.cols_feature
        col_target = self.col_target
        cols_group = self.cols_group

        if for_evaluation:  # TODO: default cols_feature
            col_target = "target"
            cols_group = ["era"]

            obj = EvaluationDataHelper.from_params(
                filename=filename, dataset_type=data_type, cols_feature=cols_feature, col_target=col_target,
                cols_group=cols_group)

        elif self.col_neutralization_reference is not None:
            obj = NeutralizationDataHelper.from_params(
                filename=filename, dataset_type=data_type, cols_feature=cols_feature, col_target=col_target,
                cols_group=cols_group, col_neutralization_reference=self.col_neutralization_reference,
                neutralization_proportion=self.neutralization_proportion)

        else:  # TODO: fix logics with evaluate_with_y
            obj = EvaluationDataHelper.from_params(
                filename=filename, dataset_type=data_type, cols_feature=cols_feature, col_target=col_target,
                cols_group=cols_group)

        if not preload:
            return obj

        return obj.load()
