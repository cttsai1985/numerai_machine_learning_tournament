import sys
import os
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, Any, Dict, List, Tuple

from .Helper import DataHelper


class DataManager:
    def __init__(
            self, working_dir: str = "./", data_mapping: Dict[str, str] = None, col_target: str = "target",
            cols_group: Optional[List[str]] = None, cols_feature: Optional[List[str]] = None):
        self.working_dir: str = working_dir if working_dir else "./"

        self.data_file_types: List[str] = ["training", "validation", "test", "live", "tournament"]
        self.data_file_names: List[str] = [
            "numerai_training_data.parquet", "numerai_validation_data.parquet", "numerai_test_data.parquet",
            "numerai_live_data.parquet", "numerai_tournament_data.parquet"]

        self.data_mapping = {k: v for k, v in zip(self.data_file_types, self.data_file_names)}
        if data_mapping:
            self.data_mapping = data_mapping
            self.data_file_types = list(data_mapping.keys())
            self.data_file_names = list(data_mapping.values())

        self._check_status()
        self.col_target: str = col_target
        self.cols_group: Optional[List[str]] = cols_group if cols_group else ["era"]
        self.cols_feature: Optional[List[str]] = cols_feature

    @classmethod
    def from_configs(cls, configs: "SolutionConfigs", **kwargs):
        # TODO: implemented and refactor other dependency
        return cls(
            working_dir=configs.input_data_dir, data_mapping=configs.data_mapping, col_target=configs.column_target,
            cols_group=configs.columns_group, cols_feature=configs.columns_feature)

    def _check_status(self):
        logging.info(f"found {len(self.data_mapping)} data: in {self.data_mapping.keys()}")
        for k, v in self.data_mapping.items():
            filepath = os.path.join(self.working_dir, v)
            if not (os.path.exists(filepath) and os.path.isfile(filepath)):
                logging.warning(f"mapped file: {v} (data_type: {k}) not in working directory: {self.working_dir}")
                continue

        return self

    def get_data_helper_by_type(
            self, data_type: str = "training", preload: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        if data_type not in self.data_mapping.keys():
            raise ValueError(f"{data_type} not in data mapping")

        obj = DataHelper.from_params(
            filename=os.path.join(self.working_dir, self.data_mapping.get(data_type)), dataset_type=data_type,
            cols_feature=self.cols_feature, col_target=self.col_target, cols_group=self.cols_group)

        if not preload:
            return obj

        return obj.load()
