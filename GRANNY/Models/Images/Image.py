from abc import ABC, abstractmethod

import numpy as np
from GRANNY.Models.IO.ImageIO import ImageIO
from GRANNY.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class Image(ABC):
    def __init__(self, image_name: str):
        self.image: NDArray[np.uint8]
        self.image_name: str = image_name
        self.image_io: ImageIO = RGBImageFile(self.image_name)

    @abstractmethod
    def loadImage(self):
        pass

    @abstractmethod
    def saveImage(self):
        pass

    @abstractmethod
    def loadMetaData(self):
        pass

    @abstractmethod
    def saveMetaData(self):
        pass

    @abstractmethod
    def getImage(self) -> NDArray[np.uint8]:
        pass

    @abstractmethod
    def getMetaKeys(self):
        pass

    @abstractmethod
    def getValue(self):
        pass

    @abstractmethod
    def setValue(self):
        pass

    @abstractmethod
    def getSpec(self):
        pass

    @abstractmethod
    def setSpec(self):
        pass

    @abstractmethod
    def getRating(self):
        pass

    @abstractmethod
    def setRating(self):
        pass

    @abstractmethod
    def extractFeature(self, mask: NDArray[np.float16]):
        pass
