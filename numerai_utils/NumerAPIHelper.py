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

_FILENAMES: List[str] = [
    "example_predictions.csv",
    "example_predictions.parquet",
    "example_validation_predictions.csv",
    "example_validation_predictions.parquet",
    "numerai_live_data.parquet",
    "numerai_tournament_data.parquet",
    "numerai_training_data.parquet",
    "numerai_validation_data.parquet",
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

        self.filenames: List[str] = _FILENAMES

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
        return os.path.join(
            self.data_model_path,
            self.dir_path_pattern.format(object_type="models", round=self.api.get_current_round()))

    def download_latest_dataset(
            self, filenames: Optional[List[str]] = None, refresh: bool = False):

        if self.is_latest_round_identifier_current_ and not refresh:
            logging.info(f"skip download since datasets are up-to-date")
            return

        if not filenames:
            filenames = self.filenames

        Path(self.data_dir_path).mkdir(parents=True, exist_ok=True)
        for filename in filenames:

            _filepath: str = os.path.join(self.data_dir_path, filename)
            if os.path.exists(_filepath) and refresh:
                os.remove(_filepath)
            self.api.download_dataset(filename=filename, dest_path=_filepath)
            logging.info(f"\n{filename} download finished.")

        logging.info(f"\nall requested datasets updated: " + ", ".join(filenames))
        self.update_round_identifier_to_current()
        return self

    def retrieve_submission_diagnostics(self, model_id: str, max_retry: int = 10, sleep_time: int = 30) -> pd.Series:
        time.sleep(15)
        series = pd.Series(self.api.submission_status(model_id))
        for i in range(max_retry):
            # noinspection PyBroadException
            try:
                while True:
                    time.sleep(sleep_time)
                    logging.info(f"retry {i}, waiting for valid submission_status return.")
                    series = pd.Series(self.api.submission_status(model_id))
                    if not series.isna().all():
                        return series

            except Exception:
                logging.error(f"retry {i} times", exc_info=True)

        return series

    def evaluate_online_predictions(
            self, model_name: Optional[str] = None, dir_path: Optional[str] = None, refresh: bool = False):
        if not dir_path:
            dir_path = self.root_dir_path

        result_filepath = os.path.join(self.result_dir_current_round_, Path(dir_path).name + ".csv")
        if os.path.exists(result_filepath) and not refresh:
            logging.info(f"skip upload since result {result_filepath} exists and not refresh")
            return True

        # submit submission
        model_id = self.api.get_models()[model_name]  # get model_id
        logging.info(f"model w/ its model_id: {model_name} ({model_id})")
        self.api.upload_predictions(os.path.join(dir_path, "tournament_predictions.csv"), model_id=model_id)

        # retrieve submission
        series = self.retrieve_submission_diagnostics(model_id)

        # process
        metrics = series[_MetricsToQuery]
        metrics_rating = series.reindex(index=list(map(lambda x: x + "Rating", _MetricsToQuery)))
        reg = re.compile('|'.join(map(re.escape, ["Rating", ])))
        metrics_rating.index = list(map(lambda s: reg.sub('', s), metrics_rating.index.tolist()))

        df = pd.concat([metrics.rename("score"), metrics_rating.rename("rating")], axis=1)
        df.index.name = "attr"
        logging.info(f"model diagnostics: {model_name} ({model_id})\n{df}")
        df.to_csv(os.path.join(dir_path, "online_model_diagnostics.csv"), )

        Path(self.result_dir_current_round_).mkdir(parents=True, exist_ok=True)
        df.to_csv(result_filepath)
        logging.info(f"write online diagnostics to {result_filepath}")
        return self
