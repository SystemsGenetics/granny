from typing import Any, List

import cv2
import numpy as np
from Granny.Analyses.Parameter import Param
from Granny.Models.Images.Image import Image
from Granny.Models.Images.MetaData import MetaData
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.MetaDataFile import MetaDataFile
from numpy.typing import NDArray


class RGBImage(Image):
    """ """

    def __init__(self, filepath: str):
        Image.__init__(self, filepath)

    def getImage(self) -> NDArray[np.uint8]:
        return self.image

    def setImage(self, image: NDArray[np.uint8]):
        self.image = image

    def loadImage(self, image_io: ImageIO):
        self.image = image_io.loadImage()

    def saveImage(self, image_io: ImageIO, folder: str):
        image_io.saveImage(self.image, folder)

    def setMetaData(self, params: List[Param]):
        self.metadata = MetaData(MetaDataFile(""))
        self.metadata.setMetaData(params)

    def loadMetaData(self):
        pass

    def saveMetaData(self):
        pass

    def setSegmentationResults(self, result: Any):
        self.result = result

    def getSegmentationResults(self) -> Any:
        return self.result
