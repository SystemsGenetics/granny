from typing import Any, List

import cv2
import numpy as np
from Granny.Analyses.Values import Value
from Granny.Models.Images.Image import Image
from Granny.Models.Images.MetaData import MetaData
from Granny.Models.IO.ImageIO import ImageIO
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
        Image.__init__(self, filepath)

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

    # todo: folder should be change to file path
    def saveImage(self, image_io: ImageIO, folder: str):
        """
        {@inheritdoc}
        """
        image_io.saveImage(self.image, folder)

    def updateMetaData(self, params: List[Value]):
        """
        {@inheritdoc}
        """
        self.metadata.updateParameters(params)

    def setMetaData(self, metadata: MetaData):
        """
        {@inheritdoc}
        """
        self.metadata = metadata

    def getMetaData(self) -> MetaData:
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

    # todo: move to Granny.Analyses.Segmentation
    def extractFeature(self) -> List[Image]:
        """
        Extracts all the instances detected stored in self.result.
        """
        self.checkResult()
        # gets bounding boxes, binary masks, and the original full-tray image array
        result = self.results
        boxes: NDArray[np.float32] = result.boxes.data.numpy()
        masks: NDArray[np.float32] = result.masks.data.numpy()
        tray_image: NDArray[np.uint8] = self.getImage()

        # sorts boxes and masks based on y-coordinates
        # todo: sort them using both x and y coordinates as numbering convention
        y_order = boxes[:, 1].argsort()  # type: ignore
        sorted_boxes = boxes[y_order]  # type: ignore
        sorted_masks = masks[y_order]  # type: ignore

        # extracts instances based on bounding boxes and masks
        individual_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_instance: Image = RGBImage(self.getImageName())
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # returns a list of individual instances
        return individual_images
