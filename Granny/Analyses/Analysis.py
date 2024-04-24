from abc import ABC, abstractmethod
from typing import List

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
        self.params: List[Param] = []

    def resetParam(self):
        """
        Resets the parameter list to an empty list
        """
        self.params = []

    def addParam(self, *param: Param):
        """
        Adds a parameter to the parameter list
        """
        for p in param:
            self.params.append(p)

    def getParams(self) -> List[Param]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return list(self.params)

    @abstractmethod
    def performAnalysis(self) -> None:
        """
        Once all required paramterers have been set, this function is used
        to perform the analysis.
        """
        pass
