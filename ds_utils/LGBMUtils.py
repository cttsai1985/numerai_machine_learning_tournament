import lightgbm
import numpy as np
from typing import Tuple

from ds_utils.Metrics import spearman_corr, pearson_corr


def _cast_proba_into_label(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    if labels.shape == preds.shape:
        return preds

    preds = np.argmax(preds.reshape(labels.shape[0], preds.shape[0] // labels.shape[0]), axis=1)
    return preds


def lgbm_spearman_eval_func(preds: np.ndarray, train_data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    labels = train_data.get_label()
    preds = _cast_proba_into_label(labels, preds)
    return "neg_spearman_corr", -1. * spearman_corr(labels, preds), False


def lgbm_pearson_eval_func(preds: np.ndarray, train_data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    labels = train_data.get_label()
    preds = _cast_proba_into_label(labels, preds)
    return "neg_pearson_corr", -1. * pearson_corr(labels, preds), False


def lgbm_mae_eval_func(preds: np.ndarray, train_data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    labels = train_data.get_label()
    preds = _cast_proba_into_label(labels, preds)
    return "mean_absolute_error", np.abs((labels - preds).mean()), False
