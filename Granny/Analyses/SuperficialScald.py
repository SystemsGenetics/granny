"""
This module performs superficial scald calculation on "Granny Smith" apple image files.
The analysis is conducted as follows:
    1. It loads input images from a specified directory.
    2. It processes each image to remove unwanted areas (e.g., purple tray residues).
    3. It identifies and removes scald regions using morphological operations and color space
    analysis.
    4. It calculates the percentage of the image affected by scald.
    5. It outputs processed images and analysis results to specified directories.

date: July 12, 2024
author: Nhan H. Nguyen
"""

import os
from datetime import datetime
from multiprocessing import Pool
from typing import List, Tuple, cast

import cv2
import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.FloatValue import FloatValue
from Granny.Models.Values.ImageListValue import ImageListValue
from Granny.Models.Values.IntValue import IntValue
from Granny.Models.Values.MetaDataValue import MetaDataValue
from numpy.typing import NDArray


class SuperficialScald(Analysis):
    """
    Analysis class for detecting superficial scald regions on "Granny Smith" apple images.

    Attributes:
        __analysis_name__ (str): The name of the analysis ('scald').
        input_images (ImageListValue): List of input images to analyze.
        output_images (ImageListValue): List of output images with scald regions identified.
        output_results (MetaDataValue): Metadata value containing the directory where analysis
        results are saved.
    """
    
    __analysis_name__ = "scald"

    def __init__(self):
        super().__init__()

        # sets up input and output directory
        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)
        self.output_images = ImageListValue(
            "output", "output", "The output directory where analysis' images are written."
        )
        result_dir = os.path.join(
            os.curdir,
            "results",
            self.__analysis_name__,
            datetime.now().strftime("%Y-%m-%d-%H-%M"),
        )
        self.output_images.setValue(result_dir)
        self.addInParam(self.input_images)

        # sets up output result directory
        self.output_results = MetaDataValue(
            "results", "results", "The output directory where analysis' results are written."
        )
        self.output_results.setValue(result_dir)

    def _smoothMask(self, bin_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Smooths scald region with basic morphological operations.

        Args:
            bin_mask (NDArray[np.uint8]): Binary mask representing scald regions.

        Returns:
            NDArray[np.uint8]: Smoothed binary mask after morphological operations.
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

    def _removeScald(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Removes the scald region from individual apple images.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            Tuple[NDArray[np.uint8], NDArray[np.uint8]]: Tuple containing:
                - NDArray[np.uint8]: Binary mask of scald regions.
                - NDArray[np.uint8]: Image with scald regions removed.
        """

        # convert from BGR to Lab color space
        new_img = img.copy()
        lab_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

        def _calculate_threshold_from_hist(hist: NDArray[np.int8]) -> int:
            hist_range = 255 - (hist[::-1] != 0).argmax() - (hist != 0).argmax()
            threshold = np.max(np.argsort(hist)[-10:])
            threshold = int(threshold - 1 / 3 * hist_range)
            threshold = 100 if threshold < 100 else int(threshold)
            return threshold

        # create binary matrices
        hist, _ = np.histogram(lab_img[:, :, 1], bins=256, range=(0, 255))
        threshold_value = _calculate_threshold_from_hist(hist)
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
        th123 = self._smoothMask(th123)

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return th123, new_img

    def _removeTrayResidue(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Removes the surrounding purple tray residue from individual apples using YCrCb color space.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            NDArray[np.uint8]: Image with surrounding purple tray residue removed.
        """
        # convert BGR to YCrCb
        new_img = img.copy()
        ycc_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

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

    def _score_image(
        self, img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Cleans up individual image by removing purple tray residue and scald regions.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]: Tuple containing:
                - NDArray[np.uint8]: Image with surrounding purple tray residue removed.
                - NDArray[np.uint8]: Image with scald regions removed.
                - NDArray[np.uint8]: Binary mask of scald regions.
        """
        # removes the residue tray background
        img = self._removeTrayResidue(img)
        nopurple_img = img.copy()

        # Image smoothing
        img = cast(NDArray[np.uint8], cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0))

        # Removal of scald regions
        bw, img = self._removeScald(img)

        return nopurple_img, img, bw

    def _calculateScald(self, bw: NDArray[np.uint8], img: NDArray[np.uint8]) -> float:
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

    def _rateSuperficialScald(self, img: NDArray[np.uint8]) -> Tuple[float, NDArray[np.uint8]]:
        """
        Rates the superficial scald of the provided image array.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            Tuple[float, NDArray[np.uint8]]: Tuple containing:
                - float: Calculated rating score for the scald area.
                - NDArray[np.uint8]: Binarized image with scald regions identified.
        """
        # returns apple image with no scald
        nopurple_img, binarized_image, _ = self._score_image(img)

        # calculate the scald region and save image
        score = self._calculateScald(binarized_image, nopurple_img)
        return score, binarized_image

    def _rateImageInstance(self, image_instance: Image) -> Image:
        """
        Loads and analyzes the provided Image instance for superficial scald.

        Args:
            image_instance (Image): An instance of GRANNY.Models.Images.Image representing the image to be rated.

        Returns:
            Image: An updated Image instance containing:
                - image_name: The file name of the input image instance.
                - rating: Calculated rating for the superficial scald area.
        """
        # initiates ImageIO
        self.image_io.setFilePath(image_instance.getFilePath())

        # loads image from file system with RGBImageFile(ImageIO)
        image_instance.loadImage(image_io=self.image_io)

        # gets the image array
        img = image_instance.getImage()

        # performs superficial scald calculation
        score, binarized_image = self._rateSuperficialScald(img)

        # initiate a result Image instance with a rating and sets the NDArray to the result
        result_img: Image = RGBImage(image_instance.getImageName())
        result_img.setImage(binarized_image)

        # saves the calculated score to the image_instance as a parameter
        rating = FloatValue("rating", "rating", "Granny calculated rating of total starch area.")
        rating.setMin(0.0)
        rating.setMax(1.0)
        rating.setValue(score)

        # adds rating to result
        result_img.addValue(rating)

        return result_img

    def performAnalysis(self) -> List[Image]:
        """
        {@inheritdoc}
        """
        # initiates user's input
        self.input_images: ImageListValue = self.in_params.get(self.input_images.getName())  # type: ignore

        # initiates an ImageIO for image input/output
        self.image_io: ImageIO = RGBImageFile()

        # initiates Granny.Model.Images.Image instances for the analysis using the user's input
        self.input_images.readValue()
        self.images = self.input_images.getImageList()

        # perform analysis with multiprocessing
        num_cpu = os.cpu_count()
        cpu_count = int(num_cpu * 0.8) or 1  # type: ignore
        with Pool(cpu_count) as pool:
            results = pool.map(self._rateImageInstance, self.images)

        # adds the result list to self.output_images then writes the resulting images to folder
        self.output_images.setImageList(results)
        self.output_images.writeValue()

        # adds the result list to self.output_results then writes the resulting results to folder
        self.output_results.setImageList(results)
        self.output_results.writeValue()

        self.addRetValue(self.output_images)

        return self.output_images.getImageList()
