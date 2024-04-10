import os
from multiprocessing import Pool
from typing import Any, List, Tuple, cast

import cv2
import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Analyses.Parameter import IntParam
from Granny.Models.Images.Image import Image
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class SuperficialScald(Analysis):

    __analysis_name__ = "scald"

    def __init__(self, images: List[Image]):
        Analysis.__init__(self, images)

        # This analysis will allow the user to manually set a threshold
        # to distinguish between the brown scald regions and the green
        # peel color. By default this threshold is determined automatically
        # but we will allow the user to manually set it if they want.
        th = IntParam(
            "th", "threshold", "The green color threhsold that distinguishes non-scald regions"
        )
        th.setMin(0)
        th.setMax(255)
        self.addParam(th)

    def getParams(self) -> List[Any]:
        """
        {@inheritdoc}
        """
        return list(self.params)

    def smoothMask(self, bin_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Smooth scald region with basic morphological operations.
        By performing morphology, the binary mask will be smoothened to avoid discontinuity.
        """
        bin_mask = bin_mask

        # create a circular structuring element of size 10
        ksize = (10, 10)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        # using to structuring element to perform one close and one open operation on the binary mask
        bin_mask = cv2.dilate(
            cv2.erode(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )  # type: ignore
        bin_mask = cv2.erode(
            cv2.dilate(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )  # type: ignore
        return bin_mask

    def isolateScald(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Remove the scald region from the individual apple images.
        Note that the stem could have potentially been removed during the process.
        """
        # convert from RGB to Lab color space
        new_img = img.copy()
        lab_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_RGB2LAB))

        def calculate_threshold_from_hist(hist: NDArray[np.int8]) -> int:
            hist_range = 255 - (hist[::-1] != 0).argmax() - (hist != 0).argmax()
            threshold = np.max(np.argsort(hist)[-10:])
            threshold = int(threshold - 1 / 3 * hist_range)
            threshold = 100 if threshold < 100 else int(threshold)
            return threshold

        # create binary matrices
        hist, _ = np.histogram(lab_img[:, :, 1], bins=256, range=(0, 255))
        threshold_value = calculate_threshold_from_hist(hist)
        threshold_1 = np.logical_and((lab_img[:, :, 0] >= 1), (lab_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and(
            (lab_img[:, :, 1] >= 1), (lab_img[:, :, 1] <= threshold_value)
        )
        threshold_3 = np.logical_and((lab_img[:, :, 2] >= 1), (lab_img[:, :, 2] <= 255))

        # combine to one matrix
        th123 = np.logical_and(np.logical_and(threshold_1, threshold_2), threshold_3).astype(
            np.uint8
        )

        # perform simple morphological operation to smooth the binary mask
        th123 = self.smoothMask(th123)

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return th123, new_img

    def removeTrayResidue(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Remove the surrounding purple from the individual apples using YCrCb color space.
        This function helps remove the unwanted regions for more precise calculation of the scald area.
        """
        # convert RGB to YCrCb
        new_img = img.copy()
        ycc_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))

        # create binary matrices
        threshold_1 = np.logical_and((ycc_img[:, :, 0] >= 0), (ycc_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and((ycc_img[:, :, 1] >= 0), (ycc_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and((ycc_img[:, :, 2] >= 0), (ycc_img[:, :, 2] <= 126))

        # combine to one matrix
        th123 = np.logical_and(np.logical_and(threshold_1, threshold_2), threshold_3).astype(
            np.uint8
        )

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def score_image(
        self, img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        """
        @param img: array representation of the image

        Clean up individual image (remove purple area of the tray), and remove scald
        """
        # removes the residue tray background
        img = self.removeTrayResidue(img)
        nopurple_img = img.copy()

        # Image smoothing
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

        # Removal of scald regions
        bw, img = self.isolateScald(img)

        return nopurple_img, img, bw

    def calculateScald(self, bw: NDArray[np.uint8], img: NDArray[np.uint8]) -> float:
        """
        Calculate scald region by counting all non zeros area

        @param bw: black white binarized image with only non-scald region
        @param img: original image to be used as ground truth for scald calculation

        @return fraction: the scald region, i.e. fraction of the original image that was removed
        """
        # count non zeros of binarized image
        ground_area = 1 / 3 * np.count_nonzero(img[:, :, 0:2])

        # count non zeros of original image
        mask_area = 1 / 3 * np.count_nonzero(bw[:, :, 0:2])

        # calculate fraction
        fraction = 0
        if ground_area == 0:
            return 1
        else:
            fraction = 1 - mask_area / ground_area
        if fraction < 0:
            return 0
        return fraction

    def rateSuperficialScald(self, img: NDArray[np.uint8]) -> Tuple[float, NDArray[np.uint8]]:
        """
        Calls self.calculateScald function to calculate the scald portion of the image array.
        """
        # returns apple image with no scald
        nopurple_img, binarized_image, _ = self.score_image(img)

        # calculate the scald region and save image
        score = self.calculateScald(binarized_image, nopurple_img)
        return score, binarized_image

    def rateImageInstance(self, image_instance: Image) -> Image:
        """
        1. Loads and performs analysis on the provided Image instance.
        2. Saves the instance to result directory

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return
            image_name: file name of the image instance
            score: rating for the instance
        """
        # initiates ImageIO
        image_io = RGBImageFile(image_instance.getFilePath())

        # loads image from file system with RGBImageFile(ImageIO)
        image_instance.loadImage(image_io=image_io)

        # gets the image array
        img = image_instance.getImage()

        # performs superficial scald calculation
        score, binarized_image = self.rateSuperficialScald(img)

        # saves the output image
        image_io.saveImage(binarized_image, self.__analysis_name__)

        # image_instance.setRating(score)
        return image_instance


    def performAnalysis(self):
        """
        {@inheritdoc}
        """
        num_cpu = os.cpu_count()
        cpu_count = int(num_cpu * 0.8) or 1 # type: ignore
        with Pool(cpu_count) as pool:
            pool.map(self.rateImageInstance, self.images)
