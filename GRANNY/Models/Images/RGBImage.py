import numpy as np
from GRANNY.Models.Images.Image import Image
from GRANNY.Models.IO.ImageIO import ImageIO
from GRANNY.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class RGBImage(Image):
    def __init__(self, image_name: str):
        Image.__init__(self, image_name)

    def getImage(self) -> NDArray[np.uint8]:
        return self.image

    def loadImage(self):
        self.image = self.image_io.loadImage()

    def saveImage(self, image: NDArray[np.uint8]):
        self.image_io.saveImage(image)

    def extractFeature(self, mask: NDArray[np.float16]):
        pass

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
