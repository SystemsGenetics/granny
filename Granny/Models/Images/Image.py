import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.Values.Value import Value
from numpy.typing import NDArray


class Image(ABC):
    """
    Abstract base class for an Image module that provides a standardized interface for image
    handling, including loading, saving, and manipulating image data and metadata.

    Attributes:
        filepath (str): Absolute file path of the image file.
        results (Any): Stores segmentation results, typically of type `ultralytics.engine.results`.
        image (NDArray[np.uint8]): The image data stored as a NumPy array.
        metadata (Granny.Models.Images.MetaData): An instance of the MetaData class containing
        image metadata.
    """

    def __init__(self, filepath: str):
        """
        Initializes the Image instance with the provided file path.

        Args:
            filepath (str): The path to the image file.
        """
        self.filepath: str = os.path.abspath(filepath)
        self.results: Any = None  # type: ultralytics.engine.results.Results
        self.image: NDArray[np.uint8]
        self.metadata: Dict[str, Value] = {}

    def addValue(self, *values: Value):
        """
        Adding a new metadata value to an image.
        """
        for value in values:
            self.metadata[value.getName()] = value

    def getValue(self, key: str):
        """
        Gets the value of a single metadata value for the image.
        """
        return self.metadata[key]

    def getFilePath(self) -> str:
        """
        Returns the absolute file path of the image.

        Returns:
            str: The absolute file path.
        """
        return self.filepath

    def getImageName(self) -> str:
        """
        Returns the file name of the image.

        Returns:
            str: The file name extracted from the file path.
        """
        return Path(self.filepath).name

    @abstractmethod
    def loadImage(self, image_io: ImageIO):
        """
        Loads image data using the provided ImageIO instance.

        Args:
            image_io (ImageIO): An instance of ImageIO used to load the image.
        """
        pass

    @abstractmethod
    def saveImage(self, image_io: ImageIO, folder: str):
        """
        Saves the image data to the specified folder using the provided ImageIO instance.

        Args:
            image_io (ImageIO): An instance of ImageIO used to save the image.
            folder (str): The directory where the image will be saved.
        """
        pass

    @abstractmethod
    def getImage(self) -> NDArray[np.uint8]:
        """
        Retrieves the image data as a NumPy array.

        Returns:
            NDArray[np.uint8]: The image data.
        """
        pass

    @abstractmethod
    def setImage(self, image: NDArray[np.uint8]):
        """
        Sets the image data from a NumPy array.

        Args:
            image (NDArray[np.uint8]): The image data to set.
        """
        pass

    @abstractmethod
    def getMetaData(self) -> Dict[str, Value]:
        """
        Retrieves the metadata associated with the image.

        Returns:
            MetaData: The metadata of the image.
        """
        pass

    @abstractmethod
    def setMetaData(self, metadata: Dict[str, Value]):
        """
        Sets the metadata for the image.

        Args:
            metadata (List[Value]): The metadata to set for the image.
        """
        pass

    @abstractmethod
    def setSegmentationResults(self, results: List[Any]):
        """
        Sets the segmentation results to self.results

        Args:
            results (List[Any]): The segmentation results to set.
        """
        pass

    @abstractmethod
    def getSegmentationResults(self) -> Any:
        """
        Retrieves the segmentation results stored in the image.

        Returns:
            The segmentation results.
        """
        pass
