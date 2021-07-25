import os
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import BaseCrossValidator, PredefinedSplit

from ds_utils.DataManager import DataManager


class PredefinedSplitHelper:
    """Generate Customized Split for Cross Validation"""

    def __init__(
            self, data_manager: DataManager, cv_splitter: BaseCrossValidator, working_dir: Optional[str] = None, ):
        self.data_manager: DataManager = data_manager
        self.cv_splitter: BaseCrossValidator = cv_splitter
        self.working_dir: Optional[str] = working_dir
        self.cv_splitter_gen: BaseCrossValidator = PredefinedSplit

    @classmethod
    def from_configs(cls, configs: "SolutionConfigs") -> "PredefinedSplitHelper":
        return cls(
            data_manager=configs.data_manager_, cv_splitter=configs.template_cv_splitter_,
            working_dir=configs.input_data_dir)

    def _produce(self, data_type: Optional[str] = None) -> pd.Series:
        if data_type is None:
            data_type = "training"

        _x, _y, _group = self.data_manager.get_data_helper_by_type(data_type=data_type).data_
        arr = np.full(_x.shape[0], -1, dtype=int)
        for i, (_, v_ind) in enumerate(self.cv_splitter.split(_x, _group.astype("category").cat.codes)):
            arr[v_ind] = i

        ret = pd.Series(arr, index=_x.index, name="label_split")
        ret.to_frame().to_parquet(os.path.join(self.working_dir, f"data_split_{data_type}.parquet"))
        return ret

    def produce(self, data_type: Optional[str] = None) -> BaseCrossValidator:
        ind_split = self._produce(data_type=data_type)
        return self.cv_splitter_gen(**{"test_fold": ind_split})
