import os
from typing import cast

import cv2
import numpy as np
from Granny.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray


class RGBImageFile(ImageIO):
    """
    I/O of RGB image.
    """
    __image_type__ = "rgb"

    def __init__(self, filepath: str):
        ImageIO.__init__(self, filepath)

    def loadImage(self) -> NDArray[np.uint8]:
        """
        {@inheritdoc}
        """
        image = cv2.cvtColor(cv2.imread(self.filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cast(NDArray[np.uint8], image)

    def saveImage(self, image: NDArray[np.uint8], folder: str) -> None:
        """
        {@inheritdoc}
        """
        cv2.imwrite(os.path.join(self.image_dir, folder, self.image_name), image)

    def getType(self):
        """
        {@inheritdoc}
        """
        return RGBImageFile.__image_type__
