"""
This module performs starch analysis on cross-section user-provided image files.
The analysis is conducted as follows:
    1. parses user's input for image folder, initiates a list of Granny.Models.Images.Image,
    then runs the starch calculation on the images to get a mask and a rating for each image.
    2. overlays the mask on to the original image indicating starch clearing area of the
    cross-sections.
    3. adds the rating to the image instance
    4. outputs the masked images to image files and the ratings to a ".csv" file.

date: July 12, 2024
author: Nhan H. Nguyen
"""

import os
import yaml
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, List, Tuple, cast

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


def load_starch_scales() -> Dict[str, Dict[str, List[float]]]:
    """
    Load starch scale data from YAML asset file.

    Reads the starch_scales.yml file from Granny/assets/ directory and returns
    the variety-specific starch index and rating mappings.

    Returns:
        Dict[str, Dict[str, List[float]]]: Dictionary mapping variety names to their
            starch scales. Format: {'HONEY_CRISP': {'index': [...], 'rating': [...]}, ...}
    """
    # Get path to this file (Granny/Analyses/StarchArea.py)
    current_dir = os.path.dirname(__file__)

    # Navigate to Granny/assets/starch_scales.yml
    yaml_path = os.path.join(current_dir, '..', 'assets', 'starch_scales.yml')

    # Load and return the YAML data
    with open(yaml_path, 'r') as file:
        starch_data = yaml.safe_load(file)

    return starch_data


class StarchScales:
    """
    A class to store starch scale indices and corresponding ratings for different apple varieties.

    This class provides predefined starch index and rating values for various apple varieties
    loaded from the YAML asset file (Granny/assets/starch_scales.yml).
    These values are used to evaluate the starch content in apples, which is an indicator of
    their ripeness and suitability for consumption or storage.

    Attributes: (Loaded from starch_scales.yml)
        HONEY_CRISP (Dict[str, List[float]]): Starch index and rating for Honey Crisp apples.
        WA38_1 (Dict[str, List[float]]): Starch index and rating for WA38_1 apples.
        WA38_2 (Dict[str, List[float]]): Starch index and rating for WA38_2 apples.
        ALLAN_BROS (Dict[str, List[float]]): Starch index and rating for Allan Bros apples.
        GOLDEN_DELICIOUS (Dict[str, List[float]]): Starch index and rating for Golden Delicious apples.
        GRANNY_SMITH (Dict[str, List[float]]): Starch index and rating for Granny Smith apples.
        JONAGOLD (Dict[str, List[float]]): Starch index and rating for Jonagold apples.
        CORNELL (Dict[str, List[float]]): Starch index and rating for Cornell apples.
    """
    pass


# Load starch scales from YAML and dynamically set them as class attributes
_starch_data = load_starch_scales()
for variety_name, scale_data in _starch_data.items():
    setattr(StarchScales, variety_name, scale_data)


class StarchArea(Analysis):
    """
    This class performs starch content analysis on apple images.

    This class extends the Analysis base class and provides functionality to
    calculate starch content in apple images, rate the images based on predefined
    starch scales in StarchScales, and save the results.

    Attributes:
        images (List[Image]): List of Image objects to be analyzed.
        starch_scales (StarchScales): Reference to the StarchScales class containing starch scales data.
        input_images (ImageListValue): Input parameter for directory containing input images.
        output_images (ImageListValue): Output parameter for directory to save analyzed images.
        output_results (MetaDataValue): Output parameter for directory to save analysis results.
    """

    __analysis_name__ = "starch"

    def __init__(self):
        super().__init__()

        self.images: List[Image] = []
        self.starch_scales = StarchScales

        # sets up input and output directory
        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)

        # Starch threshold parameter
        self.starch_threshold = IntValue(
            "starch_threshold",
            "starch_threshold",
            "Threshold value for starch detection (0-255 range). This value is converted to a "
            + "percentage (value/255) and applied to each image's actual pixel range. "
            + "Pixels with gray values <= threshold percentage are considered starch. "
            + "Lower values detect only darker starch regions, higher values include lighter regions. "
            + "Default is 140 (55% of range).",
        )
        self.starch_threshold.setMin(0)
        self.starch_threshold.setMax(255)
        self.starch_threshold.setValue(140)
        self.starch_threshold.setIsRequired(False)

        # Gaussian blur kernel size parameter
        self.blur_kernel = IntValue(
            "blur_kernel",
            "blur_kernel",
            "Size of the Gaussian blur kernel for noise reduction preprocessing. "
            + "Must be an odd positive integer. Larger values produce more smoothing. "
            + "Default is 7 (creates a 7x7 kernel).",
        )
        self.blur_kernel.setMin(1)
        self.blur_kernel.setMax(99)
        self.blur_kernel.setValue(7)
        self.blur_kernel.setIsRequired(False)

        # Visualization parameter
        self.mask_alpha = FloatValue(
            "mask_alpha",
            "mask_alpha",
            "Alpha transparency value for starch mask overlay on output images. "
            + "Range is 0.0 (transparent) to 1.0 (opaque), default is 0.6.",
        )
        self.mask_alpha.setMin(0.0)
        self.mask_alpha.setMax(1.0)
        self.mask_alpha.setValue(0.6)
        self.mask_alpha.setIsRequired(False)

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
        self.addInParam(self.input_images, self.starch_threshold, self.blur_kernel, self.mask_alpha)

        # sets up output result directory
        self.output_results = MetaDataValue(
            "results",
            "results",
            "The output directory where analysis' results are written.",
        )
        self.output_results.setValue(result_dir)

    def _drawMask(self, img: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Overlays a binary mask on an image.

        @param
            - img: The input image where the mask will be applied.
            - mask: The binary mask to be overlied on the image.
        """
        result = img.copy()
        color = (0, 0, 0)
        alpha = self.mask_alpha.getValue()
        for c in range(3):
            result[:, :, c] = np.where(
                mask == 0,
                result[:, :, c] * (1 - alpha) + alpha * color[c],
                result[:, :, c],
            )
        return result

    def _calculateStarch(self, img: NDArray[np.uint8]) -> Tuple[float, NDArray[np.uint8]]:
        """
        Calculates the starch content in the given image and return the modified image.

        This function processes the input image to calculate the starch content. The process
        involves blurring the image to remove noise, converting it to grayscale, extracting
        the actual pixel range (min/max), and applying a percentage-based threshold to identify
        starch regions. The threshold value (0-255) is converted to a percentage and applied
        to each image's actual pixel range, ensuring consistent starch detection across images
        with different lighting conditions. The ratio of starch pixels to the total pixels in
        the ground truth is returned along with the modified image.

        Args:
            img (NDArray[np.uint8]): The input image as a NumPy array of type np.uint8.

        Returns:
            Tuple[float, NDArray[np.uint8]]: A tuple containing:
                - float: The ratio of starch pixels to total pixels in the ground truth.
                - NDArray[np.uint8]: The modified image with identified starch regions.
        """

        def getPixelRange(img: NDArray[np.uint8]) -> Tuple[int, int]:
            """
            Extracts minimum and maximum pixel value of an image
            """
            hist, _ = np.histogram(grayscale, bins=256, range=(0, 255))
            low = (hist != 0).argmax()
            high = 255 - (hist[::-1] != 0).argmax()
            return low, high

        def remapToRange(img: NDArray[np.uint8], lIn: int, hIn: int, lOut: int = 0, hOut: int = 255):
            """
            Adjusts the intensity values of an image I to new values. This function is equivalent
            to normalize the image pixel values to [0, 255].
            """
            # Ensure img is in the range [lIn, hIn]
            img = np.clip(img, lIn, hIn)

            # Normalize the image to the range [0, 1]
            out = (img - lIn) / (hIn - lIn)

            # Scale and shift the normalized image to the range [lOut, hOut]
            out = out * (hOut - lOut) + lOut

            return out.astype(np.uint8)

        new_img = img.copy()

        # blurs the image to remove sharp noises, then converts it to gray scale
        kernel_size = self.blur_kernel.getValue()
        img = cast(NDArray[np.uint8], cv2.GaussianBlur(img, (kernel_size, kernel_size), 0))
        grayscale = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        # extract actual min/max pixel values from the image
        low, high = getPixelRange(grayscale)

        # calculate percentage-based threshold
        # User inputs threshold in 0-255 range (e.g., 172)
        # Convert to percentage and apply to actual image range
        image_threshold = self.starch_threshold.getValue()
        threshold_percentage = image_threshold / 255.0
        threshold_value = low + (high - low) * threshold_percentage

        # create thresholded matrices using percentage-based threshold on original range
        mask = np.logical_and((grayscale > 0), (grayscale <= threshold_value)).astype(np.uint8)

        # normalize image to [0, 255] for visualization only
        grayscale_normalized = remapToRange(grayscale, low, high)

        # creates new image using threshold matrices
        new_img = self._drawMask(new_img, mask)

        ground_truth = np.count_nonzero(
            cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > 0
        )
        starch = np.count_nonzero(mask)

        return starch / ground_truth, new_img

    def _calculateIndex(self, target: float) -> Dict[str, float]:
        """
        Calculates the starch index for different apple varieties based on the closest
        rating to the target value.

        This function computes the starch index for each apple variety based on the closest
        rating to the specified target value. It utilizes predefined starch scales stored
        in `self.starch_scales`, extracting ratings and corresponding index values. The
        closest index value to the target rating is selected for each variety.

        Args:
            target (float): The target rating value to find the closest match for.

        Returns:
            Dict[str, float]: A dictionary mapping apple variety names to their calculated
            starch index values. Example: {'HONEY_CRISP': 1.0, 'GRANNY_SMITH': 1.5, ...}
        """
        # unpacks StarchScales constants as a dictionary
        scales = {
            name: value
            for name, value in vars(self.starch_scales).items()
            if not name.startswith("_")
        }
        # results to be returned is in the form of dictionary, something like this:
        # {HONEY_CRISP : 1.0, GRANNY_SMITH : 1.5,}
        results: Dict[str, float] = {}
        for name, data in scales.items():
            # rating and index list
            rating_list = data["rating"]
            index_list = data["index"]
            # difference in rating list
            diffs = [abs(i - target) for i in rating_list]
            # finds closest index according to the diffs
            closest_index = min(range(len(diffs)), key=lambda i: diffs[i])
            # adds the starch scale to the resulting dictionary
            results[name] = index_list[closest_index]
        return results

    def _processImage(self, image_instance: Image) -> Image:
        """
        Loads and analyzes the provided Image instance to calculate starch content and ratings.

        This method performs the following steps:
        1. Sets the file path for the Image instance using self.image_io.
        2. Loads the image from the file system using image_instance.loadImage().
        3. Calculates the starch percentage in the loaded image using self._calculateStarch().
        4. Creates a new result Image instance with the calculated starch areas.
        5. Saves the calculated rating score to the result Image instance as a parameter.
        6. Calculates and adds starch scale indices to the result Image instance using self._calculateIndex().

        Args:
            image_instance (Image): An instance of GRANNY.Models.Images.Image representing the image to be rated.

        Returns:
            Image: A modified Image instance containing:
                - image_name: The file name of the input image instance.
                - score: The calculated rating for the starch content in the image.
                - Additional values for each starch scale index calculated.

        Raises:
            Any specific exceptions that might be raised during image loading or processing.

        Note:
            Ensure that self.image_io and GRANNY.Models.Images.Image are correctly initialized
            and imported respectively before calling this method.
        """
        # initiates ImageIO
        self.image_io.setFilePath(image_instance.getFilePath())

        # loads image from file system with RGBImageFile(ImageIO)
        image_instance.loadImage(image_io=self.image_io)

        # gets array image
        img = image_instance.getImage()

        # performs starch percentage calculation
        score, result = self._calculateStarch(img)

        # initiate a result Image instance with a rating and sets the NDArray to the result
        result_img: Image = RGBImage(image_instance.getImageName())
        result_img.setImage(result)

        # saves the calculated score to the image_instance as a parameter
        rating = FloatValue(
            "rating", "rating", "Granny calculated rating of total starch area."
        )
        rating.setMin(0.0)
        rating.setMax(1.0)
        rating.setValue(score)

        # calculates the closest index of the image to each StarchScale
        starch_indices = self._calculateIndex(score)

        # adds each starch scale's index into the image instance
        for scale_name, index in starch_indices.items():
            card_rating = FloatValue(
                scale_name,
                scale_name,
                "Starch scale index that cross-section is classified.",
            )
            card_rating.setValue(index)
            result_img.addValue(card_rating)

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

