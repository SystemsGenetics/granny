import numpy as np
from Images.Image import Image
from numpy.typing import NDArray


class RGBImage(Image):
    def __init__(self, image: NDArray[np.uint8]) -> None:
        Image.__init__(self, image=image)

    def getRGBImage(self) -> NDArray[np.uint8]:
        return self.image
