import numpy as np

from numpy.typing import NDArray
from Images.Image import Image


class RGBImage(Image):
    def __init__(self, image: NDArray[np.uint8]) -> None:
        Image.__init__(self, image=image)

    def getRGBImage() -> None:
        return
