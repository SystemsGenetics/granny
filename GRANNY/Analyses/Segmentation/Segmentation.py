from typing import Any, List, OrderedDict

import numpy as np
from Analyses.Analysis import Analysis
from numpy.typing import NDArray


class Segmentation(Analysis):

   def __init__(self, images: NDArray[np.uint8], **kargs):
        Analysis.__init__(self, images, kargs);

        self.yolo_model = None
        self.model_dir: str = model_dir

    def performAnalysis(self) -> None:
        # something like this
        model = YoloModel(model_dir=self.model_dir)
        self.yolo_model = model.loadModel()
        model.segmentInstances()
        pass
