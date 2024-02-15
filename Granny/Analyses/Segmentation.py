from typing import List

from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self, images: List[Image], model_dir: str):
        Analysis.__init__(self, images)
        self.model_dir: str = model_dir

    def loadModel(self):
        pass

    def performAnalysis(self) -> None:
        pass
