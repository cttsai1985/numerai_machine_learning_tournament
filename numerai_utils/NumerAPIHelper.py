import os
import logging
import re
import time
from glob import glob
from typing import Optional, List
from pathlib import Path
import pandas as pd
from numerapi import NumerAPI

_MetricsToQuery = [
    "validationSharpe",
    "validationCorrelation",
    "validationFeatureNeutralMean",

    "validationStd",
    "validationMaxFeatureExposure",
    "validationMaxDrawdown",

    "validationCorrPlusMmcSharpe",
    "validationMmcMean",
    "validationCorrPlusMmcSharpeDiff",
    "corrWithExamplePreds",
]


class NumerAPIHelper:
    def __init__(
            self, root_dir_path: Optional[str] = None, api: Optional[NumerAPI] = None):
        # TODO: wrap this into a json to init
        self.api: NumerAPI = NumerAPI() if api is None else api

        self.root_dir_path: str = os.path.join("..", "input", "numerai_tournament_resource")
        if root_dir_path:
            self.root_dir_path = root_dir_path

        logging.info(f"current root path: {self.root_dir_path}")

        self.data_dir_path: str = os.path.join(self.root_dir_path, "latest_tournament_datasets")
        logging.info(f"data dir: {self.data_dir_path}")

        self.valid_data_types: List[str] = [
            "training", "validation", "live", "test", "max_test_era", "tournament", "tournament_ids",
            "example_predictions"]

        self.current_round_str = f"{self.api.get_current_round():04d}"
        self.round_identifier_template: str = f"numerai_tournament_round"
        self.current_round_identifier: str = "_".join([self.round_identifier_template, self.current_round_str])
        logging.info(f"current round identifier: {self.current_round_identifier}")
        self.latest_round_identifier: Optional[str] = None
        self.find_latest_round_identifier()

    def find_latest_round_identifier(self, dir_path: Optional[str] = None) -> Optional[str]:
        if self.latest_round_identifier:
            return self.latest_round_identifier

        if not dir_path:
            dir_path = self.root_dir_path

        candidates = sorted(list(glob(os.path.join(dir_path, "_".join([self.round_identifier_template, "[0-9]*"])))))
        if not candidates:  # initialized
            return None

        self.latest_round_identifier = Path(candidates[-1]).stem
        return self.latest_round_identifier

    def update_round_identifier_to_current(self, dir_path: Optional[str] = None):
        if not dir_path:
            dir_path = self.root_dir_path

        file_path = os.path.join(dir_path, self.current_round_identifier)
        logging.info(f"update identifier at: {dir_path}")
        with open(file_path, "w") as f:
            pass

        self.latest_round_identifier = self.current_round_identifier
        return self

    @property
    def is_latest_round_identifier_current_(self):
        return self.latest_round_identifier == self.current_round_identifier

    @property
    def result_dir_current_round_(self):
        return os.path.join(self.root_dir_path, "live_rounds", self.current_round_identifier, )

    @property
    def dir_model_path_current_round_(self):
        return os.path.join(self.data_model_path, self.dir_path_pattern.format(
            object_type="models", round=self.api.get_current_round()))

    def download_latest_dataset(
            self, extension: str = "parquet", valid_data_types: Optional[List[str]] = None, refresh: bool = False):

        if self.is_latest_round_identifier_current_ and not refresh:
            logging.info(f"skip download since datasets are up-to-date")
            return

        if not valid_data_types:
            valid_data_types = self.valid_data_types

        Path(self.data_dir_path).mkdir(parents=True, exist_ok=True)
        for data_type in valid_data_types:
            filename = f"numerai_{data_type}_data.{extension}"
            self.api.download_latest_data(
                data_type=data_type, extension="parquet", dest_path=self.data_dir_path, dest_filename=filename)
            logging.info(f"\n{filename} download finished.")

        logging.info(f"\nall requested datasets updated: " + ",".join(valid_data_types))
        self.update_round_identifier_to_current()
        return self

    def evaluate_online_predictions(self, model_name: Optional[str] = None, dir_path: Optional[str] = None):
        if not dir_path:
            dir_path = self.root_dir_path

        model_id = self.api.get_models()[model_name]
        logging.info(f"model: {model_name} ({model_id})")
        self.api.upload_predictions(os.path.join(dir_path, "tournament_predictions.csv"), model_id=model_id)

        for i in range(10):
            try:
                time.sleep(5)
                series = pd.Series(self.api.submission_status(model_id))

            except Exception:
                logging.error(f"retry {i}", exc_info=True)
                continue

            break

        metrics = series[_MetricsToQuery]
        reg = re.compile('|'.join(map(re.escape, ["Rating", ])))
        metrics_rating = list(map(
            lambda s: reg.sub('', s), filter(lambda x: x.endswith("Rating"), series.index.tolist())))
        metrics_rating = series[metrics_rating]

        df = pd.concat([metrics.rename("score"), metrics_rating.rename("rating")], axis=1)
        df.index.name = "attr"
        logging.info(f"model diagnostics: {model_name} ({model_id})\n{df}")
        df.to_csv(os.path.join(dir_path, "online_model_diagnostics.csv"), )

        Path(self.result_dir_current_round_).mkdir(parents=True, exist_ok=True)
        result_filepath = os.path.join(self.result_dir_current_round_, Path(dir_path).name + ".csv")
        df.to_csv(result_filepath)
        logging.info(f"write online diagnostics to {result_filepath}")
        return self
