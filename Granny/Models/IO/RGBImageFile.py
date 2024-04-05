import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray


class RGBImageFile(ImageIO):
    __file_type__ = "rgb"

    def __init__(self, filepath: str):
        ImageIO.__init__(self, filepath)
        self.image: NDArray[np.uint8]
        self.file_format = ""

    def getImage(self) -> NDArray[np.uint8]:
        return self.image

    def loadImage(self):
        self.image = cast(NDArray[np.uint8], cv2.imread(self.filepath))

    def saveImage(self, folder: str) -> None:
        print(os.path.join(self.image_dir, folder, self.image_name))
        cv2.imwrite(os.path.join(self.image_dir, folder, self.image_name), self.image)

    def getType(self):
        return RGBImageFile.__file_type__
