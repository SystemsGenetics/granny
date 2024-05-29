import os
import pathlib
from typing import Any, List

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self, images: List[Image]):
        Analysis.__init__(self, images)
        self.model_dir = os.path.join(
            f"{pathlib.Path(__file__).parent}", "config", "Segmentation", "granny_v1_yolo8.pt"
        )
        self.AIModel: AIModel = YoloModel(self.model_dir)
        # loads segmentation model
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

    def segmentInstances(self, image: NDArray[np.uint8]) -> List[Any]:
        """
        Uses instance segmentation model to predict instances in the image. Instances could be
        tray_info, apples, pears, cross-sections, etc.

        1. Loads the Segmentation model (Yolov8 Segmentation)
        2. Performs instance segmentaion to find fruit and tray information
        3. Returns a list of instance of ultralytics.engine.results.Results

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return resuts: List of ultralytics.engine.results.Results of segmentation results,
        including: masks, boxes, xyxy's, classes, confident scores
        """
        # detects instances on the image
        results = self.segmentation_model.predict(image, retina_masks=True)  # type: ignore

        return results

    def performAnalysis(self) -> None:
        """
        {@inheritdoc}
        """
        # performs segmentation on each image one-by-one
        image_instances: List[Image] = []
        for image in self.images:
            # initiates ImageIO
            image_io = RGBImageFile(image.getFilePath())

            # loads image from file system with RGBImageFile(ImageIO)
            image.loadImage(image_io=image_io)

            # predicts fruit instances in the image
            result = self.segmentInstances(image.getImage())

            # sets segmentation result
            image.setSegmentationResults(results=result)
            image_instances.append(image)

        # replaces the image list with the images containing the segmentation result
        self.images = image_instances
