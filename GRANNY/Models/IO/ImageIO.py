from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ImageIO(ABC):
    def __init__(self, filepath: str):
        self.filepath: str = filepath

    @abstractmethod
    def saveImage(self, image: NDArray[np.uint8], format: str):
        pass

    @abstractmethod
    def loadImage(self) -> NDArray[np.uint8]:
        pass

    @abstractmethod
    def getType(self):
        pass
