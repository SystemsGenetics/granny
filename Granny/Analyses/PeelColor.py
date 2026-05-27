"""
This module performs color extraction/evaluation calculation on pear image files.
The analysis is conducted as follows:
    1. It loads input images from a specified directory.
    2. Removes surrounding purple from apples using YCrCb color space.
    3. Calculates mean values of green and yellow in CIELAB color space, normalized to L = 50.
    4. Calculates distance from normalized LAB to each bin color.
    5. Calculates distance to the least-mean-square line in LAB color space.

date: July 12, 2024
author: Nhan H. Nguyen
"""

import os
from datetime import datetime
from multiprocessing import Pool
from typing import Any, List, Tuple, cast

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


class PeelColor(Analysis):
    """
    Analysis class for evaluating peel color characteristics of images.

    Attributes:
        __analysis_name__ (str): Name of the analysis.
        input_images (ImageListValue): Input images directory.
        output_images (ImageListValue): Output images directory for analyzed images.
        output_results (MetaDataValue): Output directory for analysis results.
        MEAN_VALUES_A (List[float]): Mean values for A component in color card.
        MEAN_VALUES_B (List[float]): Mean values for B component in color card.
        SCORE (List[float]): Scores corresponding to color bins.
        LINE_POINT_1 (NDArray[np.float16]): First point of the reference line in LAB color space.
        LINE_POINT_2 (NDArray[np.float16]): Second point of the reference line in LAB color space.
    """

    __analysis_name__ = "color"

    def __init__(self):
        super().__init__()
        # sets up input and output directory
        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)

        # Purple removal threshold parameter
        self.purple_threshold = IntValue(
            "purple_threshold",
            "purple_threshold",
            "Threshold for removing purple background/tray pixels using YCrCb color space. "
            + "Pixels with Cb channel <= this value are kept. Range is 0 to 255, default is 126.",
        )
        self.purple_threshold.setMin(0)
        self.purple_threshold.setMax(255)
        self.purple_threshold.setValue(126)
        self.purple_threshold.setIsRequired(False)

        # Lightness minimum parameter
        self.lightness_min = IntValue(
            "lightness_min",
            "lightness_min",
            "Minimum lightness value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 0.",
        )
        self.lightness_min.setMin(0)
        self.lightness_min.setMax(255)
        self.lightness_min.setValue(0)
        self.lightness_min.setIsRequired(False)

        # Lightness maximum parameter
        self.lightness_max = IntValue(
            "lightness_max",
            "lightness_max",
            "Maximum lightness value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 255.",
        )
        self.lightness_max.setMin(0)
        self.lightness_max.setMax(255)
        self.lightness_max.setValue(255)
        self.lightness_max.setIsRequired(False)

        # Green channel minimum parameter
        self.green_min = IntValue(
            "green_min",
            "green_min",
            "Minimum green channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 0.",
        )
        self.green_min.setMin(0)
        self.green_min.setMax(255)
        self.green_min.setValue(0)
        self.green_min.setIsRequired(False)

        # Green channel maximum parameter
        self.green_max = IntValue(
            "green_max",
            "green_max",
            "Maximum green channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 128.",
        )
        self.green_max.setMin(0)
        self.green_max.setMax(255)
        self.green_max.setValue(128)
        self.green_max.setIsRequired(False)

        # Yellow channel minimum parameter
        self.yellow_min = IntValue(
            "yellow_min",
            "yellow_min",
            "Minimum yellow channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 128.",
        )
        self.yellow_min.setMin(0)
        self.yellow_min.setMax(255)
        self.yellow_min.setValue(128)
        self.yellow_min.setIsRequired(False)

        # Yellow channel maximum parameter
        self.yellow_max = IntValue(
            "yellow_max",
            "yellow_max",
            "Maximum yellow channel value for peel color detection in LAB color space. "
            + "Range is 0 to 255, default is 255.",
        )
        self.yellow_max.setMin(0)
        self.yellow_max.setMax(255)
        self.yellow_max.setValue(255)
        self.yellow_max.setIsRequired(False)

        # Normalization lightness parameter
        self.normalize_lightness = IntValue(
            "normalize_lightness",
            "normalize_lightness",
            "Target lightness value for color normalization in LAB space. "
            + "Range is 0 to 100, default is 50.",
        )
        self.normalize_lightness.setMin(0)
        self.normalize_lightness.setMax(100)
        self.normalize_lightness.setValue(50)
        self.normalize_lightness.setIsRequired(False)

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
        self.addInParam(
            self.input_images,
            self.purple_threshold,
            self.lightness_min,
            self.lightness_max,
            self.green_min,
            self.green_max,
            self.yellow_min,
            self.yellow_max,
            self.normalize_lightness,
        )

        # sets up output result directory
        self.output_results = MetaDataValue(
            "results",
            "results",
            "The output directory where analysis' results are written.",
        )
        self.output_results.setValue(result_dir)
        # values of the color cards normalized to the LMS line
        self.MEAN_VALUES_A: List[float] = [
            -36.64082458,
            -35.82390694,
            -29.47956688,
            -24.68504792,
            -21.51960279,
            -21.49440178,
            -19.49577289,
            -16.92159296,
            -13.70076143,
            -13.34873991,
        ]
        self.MEAN_VALUES_B: List[float] = [
            57.4946451,
            58.6671866,
            67.77337014,
            74.65505828,
            79.19849765,
            79.23466925,
            82.10334927,
            85.79813151,
            90.42106829,
            90.92633324,
        ]
        self.SCORE: List[float] = [
            0.5192001330394723,
            0.5233446426876467,
            0.5838859128997311,
            0.6529992071684837,
            0.6934065834794143,
            0.7210722038415041,
            0.7302285740121869,
            0.7652909029091124,
            0.8066780913327652,
            0.8106974639404376,
        ]

        self.LINE_POINT_1: NDArray[np.float16] = np.array(
            [-76.69774, 0.0], dtype=np.float16
        )
        self.LINE_POINT_2: NDArray[np.float16] = np.array(
            [0.0, 110.0861], dtype=np.float16
        )

    def remove_purple(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Remove surrounding purple from individual apples using YCrCb color space.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            NDArray[np.uint8]: Processed image array with purple regions removed.
        """
        # convert BGR to YCrCb
        new_img = img.copy()
        ycc_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))

        # create binary matrices
        threshold_1 = np.logical_and((ycc_img[:, :, 0] >= 0), (ycc_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and((ycc_img[:, :, 1] >= 0), (ycc_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and((ycc_img[:, :, 2] >= 0), (ycc_img[:, :, 2] <= self.purple_threshold.getValue()))

        # combine to one matrix
        th123 = np.logical_and(
            np.logical_and(threshold_1, threshold_2), threshold_3
        ).astype(np.uint8)

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def get_green_yellow_values(
        self, img: NDArray[np.uint8]
    ) -> Tuple[float, float, float]:
        """
        Get mean pixel values representing green and yellow in CIELAB color space, normalized to L = 50.

        Args:
            img (NDArray[np.uint8]): Original BGR image array.

        Returns:
            Tuple[float, float, float]: Mean values for L, A, B channels in LAB color space.
        """
        # convert from BGR to Lab color space
        lab_img = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

        # create binary matrices
        threshold_1 = np.logical_and((lab_img[:, :, 0] > self.lightness_min.getValue()), (lab_img[:, :, 0] < self.lightness_max.getValue()))
        threshold_2 = np.logical_and((lab_img[:, :, 1] > self.green_min.getValue()), (lab_img[:, :, 1] < self.green_max.getValue()))
        threshold_3 = np.logical_and((lab_img[:, :, 2] > self.yellow_min.getValue()), (lab_img[:, :, 2] < self.yellow_max.getValue()))

        # combine to one matrix
        th123 = np.logical_and(
            np.logical_and(threshold_1, threshold_2), threshold_3
        ).astype(np.uint8)

        # apply the binary mask on the image
        for i in range(3):
            lab_img[:, :, i] = lab_img[:, :, i] * th123

        # get mean values from each channel
        pixel_count = np.count_nonzero(th123)
        if pixel_count == 0:
            return (float('nan'), float('nan'), float('nan'))

        mean_l = np.sum(lab_img[:, :, 0]) / pixel_count * 100 / 255
        mean_a = np.sum(lab_img[:, :, 1]) / pixel_count - 128
        mean_b = np.sum(lab_img[:, :, 2]) / pixel_count - 128

        # normalize by shifting point in the spherical coordinates
        radius = np.sqrt(mean_l**2 + mean_a**2 + mean_b**2)
        scaled_l = self.normalize_lightness.getValue()
        scaled_a = np.sign(mean_a) * np.sqrt(
            np.abs(radius**2 - scaled_l**2) / (1 + (mean_b / mean_a) ** 2)
        )
        scaled_b = np.sign(mean_b) * mean_b / mean_a * scaled_a

        return (scaled_l, scaled_a, scaled_b)

    def calculate_bin_distance(
        self, color_list: List[float], method: str = "Euclidean"
    ) -> Tuple[int, NDArray[np.float16]]:
        """
        Calculate distance from normalized LAB color to each bin color.

        Args:
            color_list (List[float]): List containing color values.
            method (str, optional): Method for distance calculation. Defaults to "Euclidean".

        Returns:
            Tuple[int, NDArray[np.float16]]: Bin number and distance array.
        """
        bin_num = 0
        dist: NDArray[np.float16]
        if method == "Euclidean":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt(
                (dist_a / np.linalg.norm(dist_a)) ** 2
                + (dist_b / np.linalg.norm(dist_b)) ** 2
            )
            bin_num = np.argmin(dist) + 1
        if method == "X-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_a / np.linalg.norm(dist_a)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Y-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_b / np.linalg.norm(dist_b)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Score":
            dist = color_list[0] - np.array(self.SCORE)
            dist = np.abs(dist)
            bin_num = np.argmin(dist) + 1
        return bin_num, dist

    def _calculate_score_distance(
        self, color_list: List[float]
    ) -> Tuple[Tuple[float, float], float, float, float]:
        """
        Calculate distance to least-mean-square line in LAB color space.

        Args:
            color_list (List[float]): List containing color values.

        Returns:
            Tuple[Tuple[float, float], float, float, float]: Projection, score, distance, point.
        """

        def calculate_intersection(
            line1: Tuple[Any, Any],
            line2: Tuple[Any, Any],
        ) -> Tuple[float, float]:
            """Calculates the intersection of two lines, each line is presented by 2 coordinates point."""

            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a: Tuple[Any, Any], b: Tuple[Any, Any]):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)

            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return (x, y)

        score = 0
        distance = 0
        point = 0
        color_point = np.array([color_list[1], color_list[2]]).astype(dtype=float)
        n = self.LINE_POINT_2 - self.LINE_POINT_1
        n /= np.linalg.norm(n)
        projection = calculate_intersection(
            ((0.0, 0.0), (color_list[1], color_list[2])),
            (self.LINE_POINT_1, self.LINE_POINT_2),
        )
        score = cast(
            float,
            np.linalg.norm(projection - self.LINE_POINT_1)
            / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1),
        )
        distance = cast(
            float,
            np.linalg.norm(
                np.cross(
                    self.LINE_POINT_2 - self.LINE_POINT_1,
                    color_point - self.LINE_POINT_1,
                )
            )
            / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1),
        )
        point = np.sign(color_point[1] - projection[1])
        if score < 0:
            score = float(0)
        elif score > 1:
            score = float(1.0)
        return projection, score, distance, point

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

        # gets the image array
        img = image_instance.getImage()

        # remove surrounding purple
        img = self.remove_purple(img)

        # image smoothing
        img = cast(NDArray[np.uint8], cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0))

        # get image values
        l, a, b = self.get_green_yellow_values(img)

        # calculate distance to the least-mean-square line
        (
            projection,
            score,
            orth_distance,
            point,
        ) = self._calculate_score_distance([l, a, b])

        # bin number according to the color card
        bin_num, _ = self.calculate_bin_distance([score], method="Score")
        bin_value = FloatValue(
            "bin",
            "bin_num",
            "Granny sorted bin number of the peel color, according to the color card.",
        )
        bin_value.setMin(0.0)
        bin_value.setMax(1.0)
        bin_value.setValue(bin_num)

        # score
        score_value = FloatValue(
            "score", "score", "Granny calculated rating of the peel color."
        )
        score_value.setMin(0.0)
        score_value.setMax(1.0)
        score_value.setValue(score)

        # distance
        distance_value = FloatValue(
            "distance",
            "distance",
            "Granny calculated distance from the LMS best-fit line.",
        )
        distance_value.setValue(orth_distance)

        # relative location value to the LMS fit line,
        # i.e. 1:above or -1:below
        location_value = FloatValue(
            "location",
            "location",
            "Granny calculated location wrt. the LMS best-fit line.",
        )
        location_value.setValue(point)

        # LAB color space
        l_value = IntValue(
            "l", "L", "Granny calculated L value of the image in the LAB space."
        )
        l_value.setValue(l)
        a_value = IntValue(
            "a", "A", "Granny calculated A value of the image in the LAB space."
        )
        a_value.setValue(a)
        b_value = IntValue(
            "b", "B", "Granny calculated B value of the image in the LAB space."
        )
        b_value.setValue(b)

        # adds ratings to  to the image_instance as parameters
        image_instance.addValue(
            bin_value,
            score_value,
            distance_value,
            location_value,
            l_value,
            a_value,
            b_value,
        )

        return image_instance

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
