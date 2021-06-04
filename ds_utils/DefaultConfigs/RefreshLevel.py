import logging
from typing import List, Tuple

_PreExistedLevel: List[Tuple[str, int]] = [
    ("data", 0), ("hyper_parameters", 1), ("model", 2), ("model_refit", 3), ("predictions", 4)]


class RefreshLevel:
    @classmethod
    def possible_refresh_levels(cls):
        return _PreExistedLevel

    def __init__(self, level_name: str = "predictions"):
        self.level_name: str = level_name
        self.level_value: int = -1
        for lvl_name, lvl in _PreExistedLevel:
            if self.level_name == lvl_name:
                self.level_value = lvl

        if self.level_value == -1:
            logging.warning(f"{level_name} not in pre-existed levels")

    def __lt__(self, other) -> bool:
        return self.level_value < other.level_value

    def __le__(self, other):
        return self.level_value <= other.level_value

    def __eq__(self, other):
        return self.level_value == other.level_value

    def __ne__(self, other):
        return self.level_value != other.level_value

    def __gt__(self, other):
        return self.level_value > other.level_value

    def __ge__(self, other):
        return self.level_value >= other.level_value


if "__main__" == __name__:
    print(RefreshLevel("predictions") > RefreshLevel("model"))
    print(RefreshLevel("predictions") <= RefreshLevel("model"))
    print(RefreshLevel("model") == RefreshLevel("model"))
