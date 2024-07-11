from typing import Any, Dict, cast

import cv2
import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.Values.Value import Value
from numpy.typing import NDArray


class RGBImage(Image):
    """
    An implementation of the Image class for handling RGB images.

    Provides methods for loading, saving, and manipulating RGB image data and metadata.

    Attributes:
        filepath (str): Absolute file path of the image file inherited from Image.
        results (Any): Stores segmentation results.
        image (NDArray[np.uint8]): The RGB image data stored as a NumPy array.
        metadata (MetaData): An instance of the MetaData class for image metadata.
    """

    def __init__(self, filepath: str):
        super().__init__(filepath)

    def rotateImage(self):
        """
        {@inheritdoc}
        """
        self.image = cast(NDArray[np.uint8], cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE))

    def getImage(self) -> NDArray[np.uint8]:
        """
        {@inheritdoc}
        """
        return self.image

    def setImage(self, image: NDArray[np.uint8]):
        """
        {@inheritdoc}
        """
        self.image = image

    def loadImage(self, image_io: ImageIO):
        """
        {@inheritdoc}
        """
        self.image = image_io.loadImage()

    def saveImage(self, image_io: ImageIO, folder: str):
        """
        {@inheritdoc}
        """
        image_io.saveImage(self.image, folder)

    def toRGB(self):
        """
        Converts BGR image to the RGB format for input
        """
        self.image = cast(NDArray[np.uint8], cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

    def toBGR(self):
        """
        Converts RGB image to the BGR format for output
        """
        self.image = cast(NDArray[np.uint8], cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

    def setMetaData(self, metadata: Dict[str, Value]):
        """
        {@inheritdoc}
        """
        for value in metadata.values():
            self.metadata[value.getName()] = value

    def getMetaData(self) -> Dict[str, Value]:
        """
        {@inheritdoc}
        """
        return self.metadata

    def setSegmentationResults(self, results: Any):
        """
        {@inheritdoc}
        """
        self.results = results

    def getSegmentationResults(self) -> Any:
        """
        {@inheritdoc}
        """
        return self.results

    def checkResult(self):
        """
        Checks if the segmentation results are present in the instance. If not then throw an error.
        """
        if self.getSegmentationResults() is None:
            ModuleNotFoundError(
                "No mask detected. Follow the instructions to perform segmentation first."
            )
