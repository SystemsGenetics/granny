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
from Granny.Models.Values.StringValue import StringValue
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
        """
        Initializes the BlushColor analysis with default parameters and output directories.
        Sets up input/output parameters, default blush threshold, and links all
        necessary values to the analysis framework.
        """ 
        super().__init__()

        self.images: List[Image] = []

        # sets up input and output directory
        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)
        self.output_images = ImageListValue(
            "output",
            "output",
            "The output directory where analysis' images are written.",
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
            "results",
            "results",
            "The output directory where analysis' results are written.",
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

        # Fruit detection threshold parameter (B channel in LAB space)
        self.fruit_threshold = IntValue(
            "fruit_threshold",
            "fruit_threshold",
            "Threshold for fruit pixel detection using the B channel in LAB color space. "
            + "Pixels with B channel values > this threshold are considered fruit. "
            + "Range is 0 to 255, default is 140.",
        )
        self.fruit_threshold.setMin(0)
        self.fruit_threshold.setMax(255)
        self.fruit_threshold.setValue(140)
        self.fruit_threshold.setIsRequired(False)

        # Visualization parameters
        self.blush_color_r = IntValue(
            "blush_color_r",
            "blush_color_r",
            "Red component of blush mask color (BGR format). Range is 0 to 255, default is 150.",
        )
        self.blush_color_r.setMin(0)
        self.blush_color_r.setMax(255)
        self.blush_color_r.setValue(150)
        self.blush_color_r.setIsRequired(False)

        self.blush_color_g = IntValue(
            "blush_color_g",
            "blush_color_g",
            "Green component of blush mask color (BGR format). Range is 0 to 255, default is 55.",
        )
        self.blush_color_g.setMin(0)
        self.blush_color_g.setMax(255)
        self.blush_color_g.setValue(55)
        self.blush_color_g.setIsRequired(False)

        self.blush_color_b = IntValue(
            "blush_color_b",
            "blush_color_b",
            "Blue component of blush mask color (BGR format). Range is 0 to 255, default is 50.",
        )
        self.blush_color_b.setMin(0)
        self.blush_color_b.setMax(255)
        self.blush_color_b.setValue(50)
        self.blush_color_b.setIsRequired(False)

        self.text_x = IntValue(
            "text_x",
            "text_x",
            "X coordinate for text position on output image. Default is 20.",
        )
        self.text_x.setMin(0)
        self.text_x.setMax(5000)
        self.text_x.setValue(20)
        self.text_x.setIsRequired(False)

        self.text_y = IntValue(
            "text_y",
            "text_y",
            "Y coordinate for text position on output image. Default is 50.",
        )
        self.text_y.setMin(0)
        self.text_y.setMax(5000)
        self.text_y.setValue(50)
        self.text_y.setIsRequired(False)

        self.font_scale = FloatValue(
            "font_scale",
            "font_scale",
            "Font scale for text labels on output images. Default is 1.0.",
        )
        self.font_scale.setMin(0.1)
        self.font_scale.setMax(10.0)
        self.font_scale.setValue(1.0)
        self.font_scale.setIsRequired(False)

        self.text_thickness = IntValue(
            "text_thickness",
            "text_thickness",
            "Thickness of text labels in pixels. Default is 3.",
        )
        self.text_thickness.setMin(1)
        self.text_thickness.setMax(50)
        self.text_thickness.setValue(3)
        self.text_thickness.setIsRequired(False)

        # adds thresholds to the parameter input list
        self.addInParam(
            self.threshold,
            self.fruit_threshold,
            self.blush_color_r,
            self.blush_color_g,
            self.blush_color_b,
            self.text_x,
            self.text_y,
            self.font_scale,
            self.text_thickness,
        )

    def _calculateBlush(
        self, img: NDArray[np.uint8]
    ) -> Tuple[float, NDArray[np.uint8]]:
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
        fruit_px = lab_img[:, :, 2] > self.fruit_threshold.getValue()
        blush_px = lab_img[:, :, 1] > blush_threshold
        new_img[:, :, 0][blush_px] = self.blush_color_r.getValue()
        new_img[:, :, 1][blush_px] = self.blush_color_g.getValue()
        new_img[:, :, 2][blush_px] = self.blush_color_b.getValue()
        blush_pct = 100 * blush_px.sum() / fruit_px.sum()

        cv2.putText(
            new_img,
            "Blush: " + str(blush_pct.round(1)) + "%",
            (self.text_x.getValue(), self.text_y.getValue()),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=self.font_scale.getValue(),
            color=(0, 0, 255),
            thickness=self.text_thickness.getValue(),
        )

        return blush_px.sum() / fruit_px.sum(), new_img

    def _processImage(self, image_instance: Image) -> Image:
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

        # Extract and add QR/barcode metadata from filename (if present)
        self._add_qr_metadata(result_img, image_instance.getImageName())

        # saves the calculated score to the image_instance as a parameter
        rating = FloatValue(
            "rating", "rating", "Granny calculated rating of total blush area."
        )
        rating.setMin(0.0)
        rating.setMax(1.0)
        rating.setValue(score)

        # adds rating to result
        result_img.addValue(rating)

        return result_img

    def _preRun(self):
        """
        {@inheritdoc}
        """
        # initiates an ImageIO for image input/output
        self.image_io: ImageIO = RGBImageFile()

    def _postRun(self, results):
        """
        {@inheritdoc}
        """
        # adds the result list to self.output_images then writes the resulting images to folder
        self.output_images.setImageList(results)
        self.output_images.writeValue()

        # adds the result list to self.output_results then writes the resulting results to folder
        self.output_results.setImageList(results)
        self.output_results.writeValue()

        self.addRetValue(self.output_images)

        return self.output_images.getImageList()
