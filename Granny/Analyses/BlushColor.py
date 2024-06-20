from typing import Any, List, cast

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image
from numpy.typing import NDArray


class BlushColor(Analysis):

    __analysis_name__ = "blush"

    def __init__(self, images: List[Image]):
        Analysis.__init__(self, images)

    def performAnalysis(self) -> None:
        pass
