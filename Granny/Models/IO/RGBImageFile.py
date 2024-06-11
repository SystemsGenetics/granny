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

    def __init__(self):
        ImageIO.__init__(self)

    def loadImage(self) -> NDArray[np.uint8]:
        """
        {@inheritdoc}
        """
        image = cv2.cvtColor(cv2.imread(self.filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cast(NDArray[np.uint8], image)

    def saveImage(self, image: NDArray[np.uint8], output_path: str) -> None:
        """
        {@inheritdoc}
        """
        if not os.path.exists(os.path.join(output_path)):
            os.makedirs(os.path.join(output_path), exist_ok=True)
        cv2.imwrite(
            os.path.join(output_path, self.image_name),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

    def getType(self):
        """
        {@inheritdoc}
        """
        return RGBImageFile.__image_type__
