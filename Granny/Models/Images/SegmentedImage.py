from typing import List

import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class SegmentedImage(object):
    def __init__(self, image: Image):
        self.image = image
        self.checkResult()

    def checkResult(self):
        """
        Checks if the masks and boxes are present in the instance. If not then throw an error.
        """
        if self.image.getSegmentationResults() is None: #type: ignore
            ModuleNotFoundError("Call Yolo to generate masks of the image before performing segmentation.")

    def getNumFeatures(self) -> int:
        """
        Returns the number of detected instances.
        """
        return len(self.image.getSegmentationResults())

    def extractFruits(self) -> List[Image]:
        """
        Returns a list of Image instances, each instance represents a fruit.
        """
        return self.image.extractFeature()


    def extractTrayInfo(self) -> List[Image]:
        """
        Returns an Image instance containing tray information about the fruits.
        """
        pass

    def getImage(self, index: int) -> Image:

        return

    def getMask(self, index: int) -> NDArray[np.uint8]:
        return self.image.getSegmentationResults()

    def getMask(self, index: int) -> NDArray[np.uint8]:
        return self.image.getSegmentationResults()
