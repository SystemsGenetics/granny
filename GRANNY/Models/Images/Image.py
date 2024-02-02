from abc import ABC, abstractmethod

import numpy as np
from GRANNY.Models.Images.MetaData import MetaData
from GRANNY.Models.IO.ImageIO import ImageIO
from GRANNY.Models.IO.MetaDataFile import MetaDataFile
from GRANNY.Models.IO.MetaDataIO import MetaDataIO
from GRANNY.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class Image(ABC):
    __attrs__ = ["image", "metadata", "name"]

    def __init__(self, image: NDArray[np.uint8]) -> None:
        self.image: NDArray[np.uint8] = image
        self.metadata: MetaData
        self.name: str = ""
        self.image_io: ImageIO = RGBImageFile(self.name)

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
    def getImage(self):
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
    def extractFeature(self):
        pass
