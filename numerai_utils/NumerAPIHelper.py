import os
import logging
from glob import glob
from typing import Optional, List
from pathlib import Path
from numerapi import NumerAPI


class NumerAPIHelper:
    def __init__(
            self, root_dir_path: Optional[str] = None, api: Optional[NumerAPI] = None):
        # TODO: wrap this into a json to init
        self.api: NumerAPI = NumerAPI() if api is None else api

        self.root_dir_path: str = os.path.join("..", "input", "numerai_tournament_resource")
        if root_dir_path:
            self.root_dir_path = root_dir_path

        print(f"current root path: {self.root_dir_path}")

        self.data_dir_path: str = os.path.join(self.root_dir_path, "latest_tournament_datasets")
        print(f"data dir: {self.data_dir_path}")

        self.valid_data_types: List[str] = [
            "training", "validation", "live", "test", "max_test_era", "tournament", "tournament_ids",
            "example_predictions"]

        self.current_round_str = f"{self.api.get_current_round():04d}"
        self.round_identifier_template: str = f"numerai_tournament_round"
        self.round_identifier_current: str = "_".join([self.round_identifier_template, self.current_round_str])
        print(f"current round identifier: {self.round_identifier_current}")

        self.round_identifier_latest: Optional[str] = None
        self.round_identifier_latest = self.find_latest_round_identifier_

    @property
    def find_latest_round_identifier_(self, dir_path: Optional[str] = None) -> Optional[str]:
        if self.round_identifier_latest:
            return self.round_identifier_latest

        if not dir_path:
            dir_path = self.root_dir_path

        candidates = sorted(
            list(glob(os.path.join(dir_path, "_".join([self.round_identifier_template, "[0-9]*"])))))
        if not candidates:  # initialized
            return None

        self.round_identifier_latest = Path(candidates[-1]).stem
        return self.round_identifier_latest

    @property
    def dir_model_path_current_round_(self):
        return os.path.join(self.data_model_path, self.dir_path_pattern.format(
            object_type="models", round=self.api.get_current_round()))

    def update_round_identifier_to_current(self, dir_path: Optional[str] = None):
        if not dir_path:
            dir_path = self.root_dir_path

        file_path = os.path.join(dir_path, self.round_identifier_current)
        print(f"update identifier at: {dir_path}")
        with open(file_path, "w") as f:
            pass
        return self

    def download_latest_dataset(
            self, extension: str = "parquet", valid_data_types: Optional[List[str]] = None, refresh: bool = False):

        if self.round_identifier_latest == self.round_identifier_current and not refresh:
            print(f"skip download since datasets are up-to-date")
            return

        if not valid_data_types:
            valid_data_types = self.valid_data_types

        Path(self.data_dir_path).mkdir(parents=True, exist_ok=True)

        for data_type in valid_data_types:
            filename = f"numerai_{data_type}_data.{extension}"
            self.api.download_latest_data(
                data_type=data_type, extension="parquet", dest_path=self.data_dir_path, dest_filename=filename)
            print(f"\n{filename} download finished.")

        print(f"\nall requested datasets updated: {valid_data_types}")
        self.update_round_identifier_to_current()

        self.round_identifier_latest = self.round_identifier_current
        print(f"current identifier at: {self.round_identifier_latest}")
        return self
