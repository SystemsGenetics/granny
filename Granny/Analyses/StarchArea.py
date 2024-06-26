import os
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
from numpy.typing import NDArray


class StarchScales:
    HONEY_CRISP: Dict[str, List[float]] = {
        "index": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        "rating": [
            0.998998748,
            0.947464712,
            0.868898986,
            0.783941273,
            0.676589664,
            0.329929925,
            0.024131710,
        ],
    }
    WA38_1: Dict[str, List[float]] = {
        "index": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "rating": [
            0.893993948,
            0.855859903,
            0.757963861,
            0.597765822,
            0.164192649,
            0.080528335,
        ],
    }
    WA38_2: Dict[str, List[float]] = {
        "index": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        "rating": [
            0.950925926,
            0.912917454,
            0.839858059,
            0.749211356,
            0.770660718,
            0.634160550,
            0.571832210,
            0.522944438,
            0.178909419,
            0.017493382,
            0.075675075,
        ],
    }
    ALLAN_BROS: Dict[str, List[float]] = {
        "index": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "rating": [
            0.997783524,
            0.988769830,
            0.951909478,
            0.877526853,
            0.721066082,
            0.673838851,
            0.417864608,
            0.091652858,
        ],
    }
    GOLDEN_DELICIOUS: Dict[str, List[float]] = {
        "index": [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
        "rating": [
            0.998544220,
            0.981819854,
            0.974722333,
            0.902015343,
            0.893566670,
            0.784215902,
            0.780621478,
            0.607040963,
            0.717128225,
            0.485321449,
            0.279959478,
            0.068212979,
        ],
    }
    GRANNY_SMITH: Dict[str, List[float]] = {
        "index": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "rating": [
            0.920742836,
            0.890332499,
            0.808227909,
            0.721813109,
            0.595806394,
            0.278299256,
            0.104111379,
        ],
    }
    JONAGOLD: Dict[str, List[float]] = {
        "index": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "rating": [
            0.898336414,
            0.859494456,
            0.806417832,
            0.742177914,
            0.653981582,
            0.483778570,
            0.387202327,
            0.284663986,
            0.175593498,
        ],
    }
    CORNELL: Dict[str, List[float]] = {
        "index": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "rating": [
            0.990554095,
            0.915430492,
            0.822470328,
            0.726896529,
            0.610745795,
            0.338955981,
            0.150869695,
            0.041547982,
        ],
    }


class StarchArea(Analysis):

    __analysis_name__ = "starch"

    def __init__(self):
        super().__init__()

        self.images: List[Image] = []
        self.starch_scales = StarchScales()

        self.input_images = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.output_images = ImageListValue(
            "output", "output", "The output directory where analysis' images are written."
        )
        self.output_images.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                datetime.now().strftime("%Y-%m-%d-%H-%M"),
            )
        )
        self.addInParam(self.input_images)

        # sets up default threshold parameter
        self.threshold = IntValue(
            "th",
            "threshold",
            "The color threshold, acting as initial anchor, that distinguishes iodine-stained "
            + "starch regions",
        )
        self.threshold.setMin(0)
        self.threshold.setMax(255)
        self.threshold.setValue(172)

        # adds parameters for argument parser
        self.addInParam(self.threshold)

    def _drawMask(self, img: NDArray[np.uint8], mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Overlays a binary mask on an image.

        @param
            - img: The input image where the mask will be applied.
            - mask: The binary mask to be overlied on the image.
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

    def _calculateStarch(self, img: NDArray[np.uint8]) -> Tuple[float, NDArray[np.uint8]]:
        """
        Calculates the starch content in the given image and return the modified image.

        This function processes the input image to calculate the starch content. The process
        involves blurring the image to remove noise, converting it to grayscale, adjusting
        its intensity values, and creating a binary thresholded image to identify the starch
        regions. The ratio of starch pixels to the total pixels in the ground truth is
        returned along with the modified image.
        """

        def extractImage(img: NDArray[np.uint8]) -> Tuple[int, int]:
            """
            Extracts minimum and maximum pixel value of an image
            """
            hist, _ = np.histogram(gray, bins=256, range=(0, 255))
            low = (hist != 0).argmax()
            high = 255 - (hist[::-1] != 0).argmax()
            return low, high

        def adjustImage(
            img: NDArray[np.uint8], lIn: int, hIn: int, lOut: int = 0, hOut: int = 255
        ):
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
        img = cast(NDArray[np.uint8], cv2.GaussianBlur(img, (7, 7), 0))
        gray = cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        # re-adjusts the image to [0 255]
        low, high = extractImage(gray)
        gray = adjustImage(gray, low, high)

        # create thresholded matrices
        image_threshold = self.threshold.getValue()
        mask = np.logical_and((gray > 0), (gray <= image_threshold)).astype(np.uint8)

        # creates new image using threshold matrices
        new_img = self._drawMask(new_img, mask)

        ground_truth = np.count_nonzero(
            cast(NDArray[np.uint8], cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > 0
        )
        starch = np.count_nonzero(mask)

        return starch / ground_truth, new_img

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

        # performs starch percentage calculation
        score, result_img = self._calculateStarch(img)

        # saves the calculated score to the image_instance as a parameter
        rating = FloatValue("rating", "rating", "Granny calculated rating of total starch area.")
        rating.setMin(0.0)
        rating.setMax(1.0)
        rating.setValue(score)

        # initiate a result Image instance with a rating and sets the NDArray to the result
        result: Image = RGBImage(image_instance.getImageName())
        result.setImage(result_img)

        # adds rating to result
        result.addValue(rating)

        return result

    def performAnalysis(self) -> List[Image]:
        """
        {@inheritdoc}
        """
        # initiates user's input
        self.input_images: ImageListValue = self.in_params.get(self.input_images.getName())  # type: ignore
        self.threshold: IntValue = self.in_params.get(self.threshold.getName())  # type:ignore

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

        # adds the result list to self.output_images
        self.output_images.setImageList(results)

        # writes the segmented images to a folder
        self.output_images.writeValue()

        self.addRetValue(self.output_images)

        return self.output_images.getImageList()
