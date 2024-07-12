"""
This module performs pear blush calculation on pear image files.
The analysis is conducted as follows:
    1. loads input images from a specified directory.
    2. calculates the percentage and visually marks the blush regions in the input image.
    3. saves the analyzed images and results, and returns a list of Image instances.

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


class BlushColor(Analysis):
    """
    Analysis class to detect and quantify blush color regions on pear fruit images.

    Inherits from Analysis class.

    Attributes:
        images : List[Image]
            A list to store instances of Image objects for analysis.

        input_images : ImageListValue
            Input parameter representing the directory containing input images.

        output_images : ImageListValue
            Output parameter representing the directory where analyzed images are saved.

        output_results : MetaDataValue
            Output parameter representing the directory where analysis results are saved.

        threshold : IntValue
            Threshold parameter used to distinguish blush regions based on the A channel in LAB color space.
    """

    __analysis_name__ = "blush"

    def __init__(self):
        super().__init__()

        self.images: List[Image] = []

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

        # sets up default threshold parameter
        self.threshold = IntValue(
            "th",
            "threshold",
            "The color threshold, acting as initial anchor, that distinguishes the blush region "
            + "on the pear skin. The threshold is the A channel value in LAB color space, "
            + "the range is from 0 to 255, and the default value is set to 148. ",
        )
        self.threshold.setMin(0)
        self.threshold.setMax(255)
        self.threshold.setValue(148)
        self.threshold.setIsRequired(False)

        # adds threshold to the parameter input list
        self.addInParam(self.threshold)

    def _calculateBlush(self, img: NDArray[np.uint8]) -> Tuple[float, NDArray[np.uint8]]:
        """
        Calculate the percentage of blush area on the pear fruit image using LAB color space.

        Args:
            img : NDArray[np.uint8]
                The input image in BGR format.

        Returns:
            Tuple[float, NDArray[np.uint8]]:
                Tuple containing the percentage of blush area and the processed image with marked blush regions.
        """
        # convert from BGR to Lab color space
        new_img = img.copy()
        lab_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

        # create thresholded matrices
        blush_threshold = self.threshold.getValue()
        fruit_px = lab_img[:, :, 2] > 140
        blush_px = lab_img[:, :, 1] > blush_threshold
        new_img[:, :, 0][blush_px] = 150
        new_img[:, :, 1][blush_px] = 55
        new_img[:, :, 2][blush_px] = 50
        blush_pct = 100 * blush_px.sum() / fruit_px.sum()

        cv2.putText(
            new_img,
            "Blush: " + str(blush_pct.round(1)) + "%",
            (20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=3,
        )

        return blush_px.sum() / fruit_px.sum(), new_img

    def _rateImageInstance(self, image_instance: Image) -> Image:
        """
        1. Loads and performs analysis on the provided Image instance.
        2. Saves the instance to result directory

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return
            image_name: file name of the image instance
            score: rating for the instance
        """
        # initiates ImageIO
        self.image_io.setFilePath(image_instance.getFilePath())

        # loads image from file system with RGBImageFile(ImageIO)
        image_instance.loadImage(image_io=self.image_io)

        # gets array image
        img = image_instance.getImage()

        # performs blush percentage calculation
        score, result = self._calculateBlush(img)

        # initiate a result Image instance with a rating and sets the NDArray to the result
        result_img: Image = RGBImage(image_instance.getImageName())
        result_img.setImage(result)

        # saves the calculated score to the image_instance as a parameter
        rating = FloatValue("rating", "rating", "Granny calculated rating of total blush area.")
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
        # self.threshold: IntValue = self.in_params.get(self.threshol   d.getName())  # type:ignore

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
