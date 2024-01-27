import numpy as np

from numpy.typing import NDArray
from Models.IO.ImageIO import ImageIO


class RGBImageFile(ImageIO):
    __attrs__ = ["filepath"]

    def __init__(self):
        super(RGBImageFile, self).__init__()
        self.filepath: str = None

    def load() -> NDArray[np.uint8]:
        return None

    def save() -> None:
        pass
