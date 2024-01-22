import numpy as np
from abc import ABC

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


class ImageIO(object):
    def __init__(self):
        pass

    def saveImage(image: Image, format: str):
        pass

    def loadImage(format: str):
        pass

    def getType():
        pass


class RGBImage(Image):
    def __init__(self, image: NDArray[np.uint8]) -> None:
        super(RGBImage, self).__init__(image)

    def getRGBImage() -> None:
        return


class RGBImageFile(ImageIO):
    __attrs__ = ["filepath"]

    def __init__(self):
        super(RGBImageFile, self).__init__()
        self.filepath: str = None

    def load() -> NDArray[np.uint8]:
        return None

    def save() -> None:
        pass
