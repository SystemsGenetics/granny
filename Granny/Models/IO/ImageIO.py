from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ImageIO(ABC):
    """
    Abstract base class to handle input and output of images.
    """

    def __init__(self):
        """
        Initializes the ImageIO object with default attributes.

        Attributes:
            filepath (str): The path to the image file.
            image_dir (str): The directory where the image file is located.
            image_name (str): The name of the image file.
        """
        self.filepath: str = ""
        self.image_dir: str = ""
        self.image_name: str = ""

    def setFilePath(self, filepath: str):
        """
        Sets the file path for the image and extracts the directory and file name.

        Parameters:
            filepath (str): The full path to the image file.

        Updates:
            self.filepath (str): Updated with the given filepath.
            self.image_dir (str): Updated to the parent directory of the filepath.
            self.image_name (str): Updated to the name of the image file.
        """
        self.filepath: str = filepath
        self.image_dir = Path(filepath).parent.as_posix()
        self.image_name = Path(filepath).name

    @abstractmethod
    def saveImage(self, image: NDArray[np.uint8], output_path: str):
        """
        Saves the provided image to the specified output path with an analysis-specific name.

        Parameters:
            image (NDArray[np.uint8]): The image data to be saved.
            output_path (str): The directory where the image should be saved.

        Raises:
            NotImplementedError: This is an abstract method and must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def loadImage(self) -> NDArray[np.uint8]:
        """
        Loads and returns the image data from the set file path.

        Returns:
            NDArray[np.uint8]: The loaded image data.

        Raises:
            NotImplementedError: This is an abstract method and must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def getType(self) -> str:
        """
        Returns the type of the image.

        Possible types include:
            - 'rgb': For standard RGB images.
            - 'gray': For grayscale images.
            - 'hyperspectral': For hyperspectral images.
            - Others as defined by subclasses.

        Returns:
            str: The type of the image.

        Raises:
            NotImplementedError: This is an abstract method and must be implemented in subclasses.
        """
        pass
