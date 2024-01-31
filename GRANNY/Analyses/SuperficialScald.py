from typing import List

from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Models.Images.Image import Image


class SuperficialScald(Analysis):
    def __init__(self, images: List[Image]):
        Analysis.__init__(self, images)
