import numpy as np
from GRANNY.Models.IO.ImageIO import ImageIO
from GRANNY.Models.IO.RGBImageFile import RGBImageFile
from Images.Image import Image
from numpy.typing import NDArray


class RGBImage(Image):
    def __init__(self, image: NDArray[np.uint8]) -> None:
        Image.__init__(self, image=image)

    def getRGBImage(self) -> NDArray[np.uint8]:
        return self.image

    def loadImage(self):
        self.image = self.image_io.loadImage()

    def saveImage(self):
        pass

    def loadMetaData(self):
        pass

    def saveMetaData(self):
        pass

    def getImage(self):
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

    def extractFeature(self):
        pass
