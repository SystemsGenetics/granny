import os
from multiprocessing import Pool
from typing import Any, List, Set

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from numpy.typing import NDArray
from ultralytics import YOLO


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self, images: List[Image], model_dir: str):
        Analysis.__init__(self, images)
        self.AIModel: AIModel = YoloModel(model_dir)

    def detectInstances(self, images: List[NDArray[np.uint8]]) -> List[Any]:
        """
        Uses Yolo model to predict instances in the image. Instances could be tray_info, apples,
        pears, cross-sections, etc.

        @param image_name: numpy array of the image. Yolo model can also accept OpenCV,
        torch.Tensor, PIL.Image, csv, image file, and URL.
        """
        return self.AIModel.predict(images, conf = 0.25, iou = 0.7, device = self.device)  # type: ignore


    def segmentInstances(self, image_instances: List[NDArray[np.uint8]]):
        """
        1. Loads the Segmentation model (Yolov8 Segmentation)
        2. Performs instance segmentaion to find fruit and tray information
        3. Returns a list of instance of ultralytics.engine.results.Results

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return resuts: List of ultralytics.engine.results.Results of segmentation results,
        including: masks, boxes, xyxy's, classes, confident scores
        """
        # loads segmentation model
        self.AIModel.loadModel()

        # detects instances on the image
        results = self.detectInstances(image_instances)

        return results

    def performAnalysis(self) -> None:
        """
        {@inheritdoc}
        """
        # loads np.ndarray image from a list of Image objects
        image_instances: List[NDArray[np.uint8]] = []
        for image in self.images:
            image.loadImage()
            image_instances.append(image.getImage())

        # performs instance segmentation to retrieve the masks
        results = self.segmentInstances(image_instances)

        # checks for potential mismatch of results, then loops through the list of Images to save
        # the segmentaiton results
        if len(results) != len(image_instances):
            raise ValueError("Different output mask length.")
        for i, result in enumerate(results):
            self.images[i].setSegmentationResults(result = result)
