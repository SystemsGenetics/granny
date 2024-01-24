from abc import ABC
from typing import Any, List, OrderedDict

import numpy as np
from model import YoloModel
from numpy.typing import NDArray


class Analysis(ABC):
    __attrs__ = ["images", "params", "param_values", "trial_num"]

    def __init__(self, images: NDArray[np.uint8]):
        self.images: NDArray[np.uint8] = images
        self.params: OrderedDict = {
            "param_name": {
                "type": None,
                "default": None,
                "upper": None,
                "lower": None,
                "valid_values": List,
                "label": None,
                "help": None,
            }
        }
        self.param_values: OrderedDict = None
        self.trial_num: int = None

    def getParams(self):
        pass

    def setResults(self, index: int, key: str, value: Any):
        pass

    def checkParams(self):
        pass

    def setParamValue(key: str, value: str) -> None:
        pass

    def getParamValue(key: str):
        pass

    def getParamKeys() -> None:
        pass

    def performAnalysis() -> None:
        pass
