from multiprocessing import Pool
from typing import List

import cv2
from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Models.Images.Image import Image
from GRANNY.Models.Images.RGBImage import RGBImage


class StarchArea(Analysis):
    def __init__(self, images: List[Image]):
        Analysis.__init__(self, images)
