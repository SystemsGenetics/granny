from typing import Any, List

import cv2
import numpy as np
from Granny.Models.Images.Image import Image
from Granny.Models.IO.ImageIO import ImageIO
from numpy.typing import NDArray


class RGBImage(Image):
    """
    """
    def __init__(self, file_path: str):
        Image.__init__(self, file_path)

    def getImage(self) -> NDArray[np.uint8]:
        return self.image

    def setImage(self, image: NDArray[np.uint8]):
        self.image = image

    def loadImage(self, image_io: ImageIO):
        self.image = image_io.loadImage()

    def saveImage(self, image_io: ImageIO, folder: str):
        image_io.saveImage(self.image, folder)

    # todo: move to segmentedimage
    def extractFeature(self) -> List[Image]:
        """
        Extracts all the instances detected stored in self.result.
        """
        # gets bounding boxes, binary masks, and original full-tray image
        boxes: NDArray[np.float32] = self.result.boxes.data.numpy()
        masks: NDArray[np.float32] = self.result.masks.data.numpy()
        image: NDArray[np.uint8] = self.getImage()

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
            mask = cv2.resize(sorted_masks[i], (image.shape[1], image.shape[0]))  # type: ignore
            for channel in range(3):
                individual_image[:, :, channel] = image[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_instance: Image = RGBImage(self.getImageName())
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)
        return individual_images

    def loadMetaData(self):
        pass

    def saveMetaData(self):
        pass

    def getMetaKeys(self):
        pass

    def getValue(self):
        pass

    def setValue(self):
        pass

    def getSpec(self):
        pass

    def setSpec(self):
        pass

    def getRating(self):
        pass

    def setRating(self):

        pass

    def setSegmentationResults(self, result: Any):
        self.result = result

    def getSegmentationResults(self) -> Any:
        return self.result
