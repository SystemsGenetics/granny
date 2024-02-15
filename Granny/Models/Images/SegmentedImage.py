from typing import List

import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class SegmentedImage(object):
    def __init__(self, image: Image, masks: NDArray[np.uint8]):
        self.image = image
        self.masks = masks

    def getNumFeatures(self) -> int:
        return 0

    def getImage(self, index: int) -> Image:
        return

    def getMask(self, index: int) -> NDArray[np.uint8]:
        return self.masks[index]

    def saveImage(self, index: int, output: RGBImageFile):
        return

    def saveMask(self, index: int):
        return
