import json
import os
import logging
import re
import time
from glob import glob
from typing import Optional, List, Iterable, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from urllib3.exceptions import NewConnectionError
from numerapi import NumerAPI


_MetricsToQuery: List[str] = [
    # Corr
    "validationCorrSharpe",
    "validationCorrMean",
    "validationFeatureNeutralCorrMean",

    # Risk
    "validationCorrStd",
    "validationFeatureCorrMax",
    "validationMaxDrawdown",

    # MMC
    "validationCorrPlusMmcSharpe",
    "validationCorrPlusMmcMean",
    "validationCorrPlusMmcStd",
    "validationMmcSharpe",
    "validationMmcMean",
    "validationMmcStd",

    "validationCorrPlusMmcSharpeDiff",
    "examplePredsCorrMean"
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
    "features.json",
]


def retry(times: int = 3, exceptions: Tuple = (ValueError, TypeError, NewConnectionError)):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :return:
    """
    def decorator(func):
        def new_func(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)

                except exceptions:
                    logging.warning(f"Exception thrown when attempting to run {func}, attempt {attempt} of {times}")
                    attempt += 1
                    time.sleep(.5)

            return func(*args, **kwargs)
        return new_func
    return decorator


class NumerAPIHelper:
    def __init__(
            self, root_dir_path: Optional[str] = None, api: Optional[NumerAPI] = None):

        self.api: NumerAPI = api
        if api is None:
            self.api: NumerAPI = NumerAPI(
                secret_key=os.environ.get("numerapiSecret", None), public_id=os.environ.get("numerapiPublicID", None))

        self.root_dir_path = root_dir_path
        if root_dir_path is None:
            self.root_dir_path: str = os.environ.get("rootResourcePath")
        logging.info(f"current root path: {self.root_dir_path}")

        self.data_dir_path: str = os.path.join(self.root_dir_path, "latest_tournament_datasets")
        logging.info(f"data dir: {self.data_dir_path}")

        # static variables
        self.round_identifier_template: str = "numerai_tournament_round"
        self.current_round_identifier: Optional[str] = None
        self.latest_round_identifier: Optional[str] = None
        self.filenames: List[str] = _FILENAMES
        self._initialize()

    @retry(times=3)
    def _api_get_current_round(self) -> int:
        return self.api.get_current_round()

    def _initialize(self):
        # noinspection PyBroadException
        try:
            _current_round: int = self._api_get_current_round()
            self.current_round_str: str = f"{_current_round:04d}"
        except Exception:
            _identifier = self.find_latest_round_identifier()
            if _identifier:
                self.current_round_str: str = _identifier.split("_")[-1]

        self.current_round_identifier: str = "_".join([self.round_identifier_template, self.current_round_str])
        self.find_latest_round_identifier()
        logging.info(f"current round identifier: {self.current_round_identifier}")
        Path(self.result_dir_current_round_).mkdir(parents=True, exist_ok=True)
        return self

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
            return self

        if not filenames:
            filenames = self.filenames

        Path(self.data_dir_path).mkdir(parents=True, exist_ok=True)
        file_paths: Iterable[str] = list(map(lambda _f: os.path.join(self.data_dir_path, _f), filenames))

        if refresh:
            for _filepath in file_paths:
                if os.path.exists(_filepath):
                    os.remove(_filepath)
                logging.info(f"remove {_filepath}")

        for _filename, _filepath in zip(filenames, file_paths):
            self.api.download_dataset(filename=_filename, dest_path=_filepath)
            logging.info(f"\n{_filename} download finished.")

        logging.info(f"\nall requested datasets updated: " + ", ".join(filenames))
        self.update_round_identifier_to_current()
        return self

    def _compose_submit_filepath(self, filename: str, dir_path: Optional[str] = None):
        if not dir_path:
            dir_path = self.root_dir_path

        filepath = os.path.join(dir_path, filename)
        return filepath

    @retry(times=3)
    def submit_diagnostics(
            self, model_name: Optional[str] = None, dir_path: Optional[str] = None,
            filename: str = "validation_predictions.csv"):
        filepath = self._compose_submit_filepath(filename=filename, dir_path=dir_path)

        # submit submission
        model_id = self.api.get_models()[model_name]  # get model_id
        logging.info(f"model ({model_name}) w/ its model_id: ({model_id})")

        diagnostics_id = self.api.upload_diagnostics(file_path=filepath, model_id=model_id)
        logging.info(f"diagnostics of {model_name} ({model_id}): {diagnostics_id}")

        results = self.api.diagnostics(model_id=model_id, diagnostics_id=diagnostics_id)
        self.save_diagnostics(results=results, dir_path=dir_path)
        return self

    def get_diagnostics(
            self, model_name: Optional[str] = None, diagnostics_id: Optional[str] = None) -> List[Dict[str, Any]]:
        model_id = self.api.get_models()[model_name]  # get model_id
        logging.info(f"model ({model_name}) w/ its model_id: ({model_id})")
        return self.api.diagnostics(model_id=model_id, diagnostics_id=diagnostics_id)

    @staticmethod
    def _extract_all_era_info_from_json(results: List[Dict[str, Any]]) -> pd.DataFrame:
        series: Dict[str, Any] = pd.Series(results[0])

        # process
        metrics = series[_MetricsToQuery]
        metrics_rating = series.reindex(index=list(map(lambda x: x + "Rating", _MetricsToQuery)))
        reg = re.compile('|'.join(map(re.escape, ["Rating", ])))
        metrics_rating.index = list(map(lambda s: reg.sub('', s), metrics_rating.index.tolist()))

        df = pd.concat([metrics.rename("score"), metrics_rating.rename("rating")], axis=1)
        df.index.name = "attr"
        return df

    def save_diagnostics(self, results: List[Dict[str, Any]], dir_path: Optional[str] = None, ):
        result_filepath = os.path.join(self.result_dir_current_round_, Path(dir_path).name + ".json")
        with open(result_filepath, "w") as f:
            json.dump(results, fp=f)

        result_filepath = os.path.join(self.result_dir_current_round_, Path(dir_path).name + ".csv")
        df = self._extract_all_era_info_from_json(results=results)
        df.to_csv(result_filepath)
        return self

    @retry(times=3)
    def submit_predictions(
            self, model_name: Optional[str] = None, dir_path: Optional[str] = None,
            filename: str = "tournament_predictions.csv"):
        filepath = self._compose_submit_filepath(filename=filename, dir_path=dir_path)

        # submit submission
        model_id = self.api.get_models()[model_name]  # get model_id
        logging.info(f"model ({model_name}) w/ its model_id: ({model_id})")

        self.api.upload_predictions(filepath, model_id=model_id, version=2)
        logging.info(f"predictions uploaded w/ its model_id: {model_name} ({model_id})")
        return self

    def do_submission(
            self, model_name: Optional[str] = None, dir_path: Optional[str] = None,
            diagnostics_filename: str = "validation_predictions.csv",
            predictions_filename: str = "tournament_predictions.csv"):
        self.submit_diagnostics(model_name=model_name, dir_path=dir_path, filename=diagnostics_filename)

        # noinspection PyBroadException
        try:
            self.submit_predictions(model_name=model_name, dir_path=dir_path, filename=predictions_filename)
        except Exception:
            logging.info(f"submission rejected for weekly predictions from {dir_path}")

        return self
