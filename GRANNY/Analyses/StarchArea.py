import os
from multiprocessing import Pool
from typing import Any, List, cast

import cv2
import numpy as np
from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Models.Images.Image import Image
from GRANNY.Models.Images.RGBImage import RGBImage
from numpy.typing import NDArray


class StarchArea(Analysis):

    def __init__(self, image: Image):
        Analysis.__init__(self, image)

    def getParams(self):
        pass

    def setResults(self, index: int, key: str, value: Any):
        pass

    def checkParams(self):
        pass

    def setParamValue(self, key: str, value: str) -> None:
        pass

    def getParamValue(self, key: str):
        pass

    def getParamKeys(self) -> None:
        pass

    def drawMask(
        self, img: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Overlays a binary mask on an image.

        Args:
            - img: The input image where the mask will be applied.
            - mask: The binary mask to be overlied on the image ().
        """
        result = img.copy()
        color = (0, 0, 0)
        alpha = 0.6
        for c in range(3):
            result[:, :, c] = np.where(
                mask == 0,
                result[:, :, c] * (1 - alpha) + alpha * color[c],
                result[:, :, c],
            )
        return result

    def smoothMask(self, bin_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Smooth binary mask with basic morphological operations.
        By performing morphology, the binary mask will be smoothened.
        """
        bin_mask = bin_mask

        # create a circular structuring element of size 10
        ksize = (10, 10)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        # using to structuring element to perform one close and one open
        # operation on the binary mask
        bin_mask = cv2.dilate(
            cv2.erode(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )
        bin_mask = cv2.erode(
            cv2.dilate(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )
        return bin_mask


    def calculateStarch(self, img: NDArray[np.uint8]) -> float:
        new_img = img.copy()
        img = cv2.GaussianBlur(img, (11, 11), 0)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        def calculate_threshold_from_hist(hist: NDArray[np.int8]) -> int:
            histogram_sum = np.sum(hist)
            left_sum = 0

            for i, bin_value in enumerate(hist):
                left_sum += bin_value
                right_sum = histogram_sum - left_sum

                if left_sum - right_sum > 0.1 * histogram_sum:
                    if i < 160:
                        return 160
                    return i
            return -1

        # create thresholded matrices
        hist, _ = np.histogram(lab_img[:, :, 2], bins=256, range=(0, 255))
        threshold_value = calculate_threshold_from_hist(hist)

        threshold_1 = np.logical_and((lab_img[:, :, 0] > 0), (lab_img[:, :, 0] <= 205))
        threshold_2 = np.logical_and((lab_img[:, :, 1] > 0), (lab_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and(
            (lab_img[:, :, 2] > 0), (lab_img[:, :, 2] <= threshold_value)
        )

        # combine to one matrix
        th123 = np.logical_and(
            np.logical_and(threshold_1, threshold_2), threshold_3
        ).astype(np.uint8)

        # perform a simple morphological operation to smooth the binary mask
        th123 = self.smoothMask(th123)

        # create new image using threshold matrices
        new_img = self.drawMask(new_img, th123)


        ground_truth = np.count_nonzero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)
        starch = np.count_nonzero(th123)

        return starch / ground_truth


    def performAnalysis(self) -> None:
        # loads image from file system with RGBImageFile(ImageIO)
        self.image.loadImage()
        # gets array image
        img = self.image.getImage()
        # performs starch percentage calculation
        result = self.calculateStarch(img)
        print(result)
        pass
