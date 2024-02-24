from typing import Any, List

import numpy as np
from Granny.Models.Images.Image import Image
from numpy.typing import NDArray


class RGBImage(Image):
    def __init__(self, image_name: str):
        Image.__init__(self, image_name)

    def getImage(self) -> NDArray[np.uint8]:
        return self.image

    def loadImage(self):
        self.image = self.image_io.loadImage()

    def saveImage(self, image: NDArray[np.uint8], analysis: str):
        self.image_io.saveImage(image, analysis)

    def extractFeature(self) -> List[Image]:
        boxes = self.result.boxes
        masks = self.result.masks
        

    def loadMetaData(self):
        pass

    def saveMetaData(self):
        pass

    def getMetaKeys(self):
        pass

    def getValue(self):
        pass

    def setValue(self):
        pass

    def getSpec(self):
        pass

    def setSpec(self):
        pass

    def getRating(self):
        pass

    def setRating(self):
        pass

    def setSegmentationResults(self, result: Any):
        self.result = result

    def getSegmentationResults(self) -> Any:
        return self.result
