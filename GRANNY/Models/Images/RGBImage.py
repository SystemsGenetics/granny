import numpy as np
from numpy.typing import NDArray


class RGBImage(Image):
    def __init__(self, image: NDArray[np.uint8]) -> None:
        super(RGBImage, self).__init__(image)

    def getRGBImage() -> None:
        return
