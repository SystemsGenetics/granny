from abc import ABC, abstractmethod
from typing import Any, List, OrderedDict

from GRANNY.Models.Images.Image import Image


class Analysis(ABC):

    __anlaysis_name__ = "analysis"

    def __init__(self, image: Image):
        """
        Intializes an instance of an Analysis object,

        @param GRANNY.Models.Images.Image An instance of an Image object

        @return GRANNY.Analyses.Analysis.Analysis object.
        """
        self.image: Image = image
        self.params: OrderedDict[str, str] = {
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
        self.param_values: OrderedDict[str, str] = {"": ""}
        self.trial_num: int = 0

    @abstractmethod
    def getParams(self):
        pass

    @abstractmethod
    def setResults(self, index: int, key: str, value: Any):
        pass

    @abstractmethod
    def checkParams(self):
        pass

    @abstractmethod
    def setParamValue(self, key: str, value: str) -> None:
        pass

    @abstractmethod
    def getParamValue(self, key: str):
        pass

    @abstractmethod
    def getParamKeys(self) -> None:
        pass

    @abstractmethod
    def performAnalysis(self) -> None:
        """
        Performs the analysis.

        Once all required paramterers have been set, this function is used
        to perform the analysis.

        @throws Exception
        """
        pass
