from typing import Optional
import numpy as np
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples


class TimeSeriesSplitGroups(_BaseKFold):
    """
    Because the TimeSeriesSplit class in sklearn does not use groups and won't respect era boundaries, we implement
    a version that will.
    """

    def __init__(self, n_splits: int = 5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        raise NotImplementedError()

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        group_list.sort()

        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(f"Cannot have number of folds = {n_folds} greater than the number of samples: {n_groups}.")

        indices = np.arange(n_samples)
        test_size = n_groups // n_folds
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (
                indices[groups.isin(group_list[:test_start])],
                indices[groups.isin(group_list[test_start:test_start + test_size])])
