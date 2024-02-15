from typing import Any, cast

import cv2
import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image
from numpy.typing import NDArray


class BlushColor(Analysis):

    __analysis_name__ = "blush"

    def __init__(self, image: Image):
        Analysis.__init__(self, image)

    def getParams(self):
        pass

    def setResults(self, index: int, key: str, value: Any):
        pass

    def checkParams(self):
        pass

    def setParamValue(self, key: str, value: str) -> None:
        pass

    def getParamValue(self, key: str):
        pass

    def getParamKeys(self) -> None:
        pass

    def performAnalysis(self) -> None:
        pass
