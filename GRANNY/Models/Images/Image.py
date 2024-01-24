from abc import ABC

import numpy as np
from numpy.typing import NDArray


class Image(ABC):
    __attrs__ = ["image", "metadata", "name"]

    def __init__(self, image: NDArray[np.uint8]) -> None:
        self.image: NDArray[np.uint8] = None
        self.metadata = None
        self.name: str = None

    def loadImage(self):
        pass

    def saveImage():
        pass

    def loadMetaData():
        pass

    def saveMetaData():
        pass

    def getImage():
        pass
