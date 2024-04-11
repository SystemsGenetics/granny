from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class ImageIO(ABC):
    """
    Abstract base class to handle input and output of images.
    """

    def __init__(self, filepath: str):
        self.filepath: str = filepath
        self.image_dir = Path(filepath).parent
        self.image_name = Path(filepath).name

    @abstractmethod
    def saveImage(self, image: NDArray[np.uint8], folder: str):
        """ """
        pass

    @abstractmethod
    def loadImage(self) -> NDArray[np.uint8]:
        """ """
        pass

    @abstractmethod
    def getType(self) -> str:
        """
        Returns image type: rgb, gray, hyperspectral, ...
        """
        pass
