import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import numpy as np
from Granny.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray


class Image(ABC):
    """
    Abstract base class for Image module.
    """

    def __init__(self, filepath: str):
        """ """
        self.filepath: str = os.path.abspath(filepath)
        self.result: Any = None  # type: ultralytics.engine.results.Results
        self.image: NDArray[np.uint8]

    def getFilePath(self) -> str:
        """
        Returns the full file path of the image
        """
        return self.filepath

    def getImageName(self) -> str:
        """ """
        return Path(self.filepath).name

    @abstractmethod
    def loadImage(self, image_io: ImageIO):
        pass

    @abstractmethod
    def saveImage(self, image_io: ImageIO, folder: str):
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
    def setImage(self, image: NDArray[np.uint8]):
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
