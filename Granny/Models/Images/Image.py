from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class Image(ABC):
    def __init__(self, image_name: str):
        self.image: NDArray[np.uint8]
        self.image_name: str = image_name
        self.image_io: ImageIO = RGBImageFile(self.image_name)
        self.result: Any = None  # type: ultralytics.engine.results.Results

    @abstractmethod
    def loadImage(self):
        pass

    @abstractmethod
    def saveImage(self, image: NDArray[np.uint8], analysis: str):
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
    def extractFeature(self) -> List["Image"]:
        pass

    @abstractmethod
    def setSegmentationResults(self, result: List[Any]):
        pass

    @abstractmethod
    def getSegmentationResults(self) -> Any:
        pass
