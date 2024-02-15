from abc import ABC, abstractmethod
from typing import Any, List, OrderedDict

from Granny.Analyses.Parameter import Param
from Granny.Models.Images.Image import Image


class Analysis(ABC):

    __analysis_name__ = "analysis"

    def __init__(self, images: List[Image]):
        """
        Intializes an instance of an Analysis object

        @param GRANNY.Models.Images.Image An instance of an Image object

        @return GRANNY.Analyses.Analysis.Analysis object.
        """
        self.images: List[Image] = images
        self.params: list[Param] = []
        self.trial_num: int = 0

    def addParam(self, param: Param):
        self.params.append(param)

    @abstractmethod
    def getParams(self) -> List[Any]:
        """Returns to the GUI/CLI all the default parameters in self.params"""
        pass

    @abstractmethod
    def setResults(self, index: int, name: str, value: Any):
        pass

    @abstractmethod
    def checkParams(self):
        pass

    @abstractmethod
    def setParamValue(self, name: str, value: str) -> None:
        pass

    @abstractmethod
    def getParamValue(self, name: str):
        pass

    @abstractmethod
    def getParamKeys(self) -> None:
        pass

    def resetTrialNum(self) -> None:
        """
        Sets the trial number back to 0.
        """
        self.trial_num = 0

    @abstractmethod
    def performAnalysis(self) -> None:
        """
        Calls multiple CPUs to perform the analysis in parallel

        Once all required paramterers have been set, this function is used
        to perform the analysis.

        @throws Exception
        """
        pass

    @abstractmethod
    def performAnalysis_multiprocessing(self, image_instance: Image) -> None:
        """

        @param image_instance An instance of a GRANNY.Models.Images.Image object
        """
        pass
