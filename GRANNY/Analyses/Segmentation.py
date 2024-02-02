from typing import List

from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Models.Images.Image import Image


class Segmentation(Analysis):
    __name__ = "segmentation"

    def __init__(self, images: List[Image], model_dir: str):
        Analysis.__init__(self, images)
        self.model_dir: str = model_dir

    def loadModel(self):
        pass

    def performAnalysis(self) -> None:
        pass
