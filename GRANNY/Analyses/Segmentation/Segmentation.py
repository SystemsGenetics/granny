from abc import ABC
from typing import Any, List, OrderedDict

import numpy as np
from Analysis import Analysis
from model import YoloModel
from numpy.typing import NDArray


class Segmentation(Analysis):
    def __init__(self, model_dir: str):
        super(self, Segmentation).__init__()
        self.yolo_model = None
        self.model_dir: str = model_dir

    def performAnalysis(self) -> None:
        # something like this
        model = YoloModel(model_dir=self.model_dir)
        self.yolo_model = model.loadModel()
        model.segmentInstances()
        pass
