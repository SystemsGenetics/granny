from typing import Any, List

import cv2
import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class SegmentedImage(object):
    def __init__(self, image: Image):
        self.image = image
        self.checkResult()
        self.individual_images: List[Image] = []
        self.tray_info: List[Image] = []

    def checkResult(self):
        """
        Checks if the masks and boxes are present in the instance. If not then throw an error.
        """
        if self.image.getSegmentationResults() is None:  # type: ignore
            ModuleNotFoundError(
                "Call Yolo to generate masks of the image before performing segmentation."
            )

    def extractFeature(self):
        """
        Extracts all the instances detected stored in self.result.
        """
        # gets bounding boxes, binary masks, and the original full-tray image array
        result = self.image.getSegmentationResults()
        boxes: NDArray[np.float32] = result.boxes.data.numpy()
        masks: NDArray[np.float32] = result.masks.data.numpy()
        tray_image: NDArray[np.uint8] = self.image.getImage()

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
            mask = cv2.resize(sorted_masks[i], (tray_image.shape[1], tray_image.shape[0]))  # type: ignore
            for channel in range(3):
                individual_image[:, :, channel] = tray_image[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_instance: Image = RGBImage(self.image.getImageName())
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # sets individual 
        self.individual_images = individual_images

    def getNumFeatures(self) -> int:
        """
        Returns the number of detected instances.
        """
        return len(self.image.getSegmentationResults())

    # todo: split trayinfo/fruits
    def extractFruits(self) -> List[Image]:
        """
        Returns a list of Image instances, each instance represents a fruit.
        """
        return self.individual_images

    # todo: split trayinfo/fruits
    def extractTrayInfo(self) -> List[Image]:
        """
        Returns an Image instance containing tray information about the fruits.
        """
        return self.tray_info

    def getImage(self, index: int) -> Image:
        return self.individual_images[index]

    def getMask(self, index: int) -> NDArray[np.uint8]:
        return self.image.getSegmentationResults()
