import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from GRANNY.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray


class RGBImageFile(ImageIO):

    def __init__(self, filepath: str):
        ImageIO.__init__(self, filepath)
        self.image_dir = Path(filepath).parent
        self.image_name = Path(filepath).name

    def loadImage(self) -> NDArray[np.uint8]:
        return cast(NDArray[np.uint8], cv2.imread(self.filepath))

    def saveImage(self, image: NDArray[np.uint8]) -> None:
        cv2.imwrite(os.path.join(self.image_dir, "results/", self.image_name), image)

    def getType(self):
        return super().getType()
