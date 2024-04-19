import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List

import numpy as np
from Granny.Analyses.Parameter import Param
from Granny.Models.Images.MetaData import MetaData
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.MetaDataFile import MetaDataFile
from numpy.typing import NDArray


class Image(ABC):
    """
    Abstract base class for Image module.
    """

    def __init__(self, filepath: str):
        """ """
        self.filepath: str = os.path.abspath(filepath)
        self.results: Any = None  # type: ultralytics.engine.results.Results
        self.image: NDArray[np.uint8]
        self.metadata: MetaData

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
        """ """
        pass

    @abstractmethod
    def saveImage(self, image_io: ImageIO, folder: str):
        """ """
        pass

    @abstractmethod
    def getImage(self) -> NDArray[np.uint8]:
        pass

    @abstractmethod
    def setImage(self, image: NDArray[np.uint8]):
        pass

    @abstractmethod
    def setMetaData(self, metadata: MetaData):
        """
        Calls MetaData class to set up a list of parameters for the analysis
        """
        pass

    @abstractmethod
    def setSegmentationResults(self, results: List[Any]):
        """
        Sets the segmentation results to self.results
        """
        pass

    @abstractmethod
    def getSegmentationResults(self) -> Any:
        """
        Returns the segmentation results stored in self.results
        """
        pass
