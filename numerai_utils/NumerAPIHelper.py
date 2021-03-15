import os
from typing import Optional, List
from pathlib import Path
from numerapi import NumerAPI


class NumerAPIHelper:
    def __init__(
            self, data_root_path: Optional[str] = None, model_root_path: Optional[str] = None,
            api: Optional[NumerAPI] = None):
        # TODO: wrap this into a json to init
        self.api: NumerAPI = NumerAPI() if api is None else api
        self.data_root_path: str = "../input/numerai_datasets/" if not data_root_path else data_root_path
        print(f"current data root: {self.data_root_path}")
        self.model_root_path: str = "../input/numerai_models/" if not model_root_path else model_root_path
        print(f"current model root: {self.model_root_path}")
        self.valid_data_types = [
            "live", "training", "validation", "test", "max_test_era", "tournament", "tournament_ids",
            "example_predictions"]

        self.dir_path_pattern: str = "numerai_{object_type}_{round:06d}"

    @property
    def dir_data_path_current_round_(self):
        return os.path.join(self.data_root_path, self.dir_path_pattern.format(
            object_type="datasets", round=self.api.get_current_round()))

    @property
    def dir_model_path_current_round_(self):
        return os.path.join(self.data_model_path, self.dir_path_pattern.format(
            object_type="models", round=self.api.get_current_round()))

    def download_latest_dataset(
            self, extension: str = "parquet", valid_data_types: Optional[List[str]] = None, refresh: bool = False):
        dest_path = self.dir_data_path_current_round_
        Path(dest_path).mkdir(parents=True, exist_ok=True)

        if not valid_data_types:
            valid_data_types = self.valid_data_types

        for data_type in valid_data_types:
            filename = f"numerai_{data_type}_data.{extension}"
            if os.path.exists(os.path.join(dest_path, filename)) and not refresh:
                print(f"skip download {filename} to {dest_path}\n")
                continue

            filename = self.api.download_latest_data(
                data_type=data_type, extension="parquet", dest_path=dest_path, dest_filename=filename)
            print(f"\n{filename} download finished.")

        return self
