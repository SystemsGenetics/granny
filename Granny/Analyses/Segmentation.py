import os
from multiprocessing import Pool
from typing import Any, List

import cv2
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

    def detectInstances(self, image: NDArray[np.uint8]) -> List[Any]:
        """
        Uses Yolo model to predict instances in the image. Instances could be tray_info, apples,
        pears, cross-sections, etc.

        @param image_name: numpy array of the image. Yolo model can also accept OpenCV,
        torch.Tensor, PIL.Image, csv, image file, and URL.
        """
        return self.AIModel.predict(image, conf=0.25, iou=0.7, device=self.device)  # type: ignore

    def segmentInstances(self, image_instance: Image):
        """
        1. Loads the Segmentation model (Yolov8 Segmentation)
        2. Performs instance segmentaion to find fruit and tray information
        3. Returns an instance of ultralytics.engine.results.Results

        @param image_instance: An GRANNY.Models.Images.Image instance

        @return

        """
        # loads image from file system with RGBImageFile(ImageIO)
        image_instance.loadImage()

        # gets array image
        img = image_instance.getImage()

        # loads segmentation model
        self.AIModel.loadModel()

        # detects instances on the image
        results = self.detectInstances(img)

        # saves segmentation results to the Image instance
        image_instance.setSegmentationResults(results)

        # 
        individual_images: List[Image] = image_instance.extractFeature()

    def performAnalysis_multiprocessing(self, image_instance: Image) -> None:
        """
        {@inheritdoc}
        """
        self.segmentInstances(image_instance)

    def performAnalysis(self) -> None:
        """
        {@inheritdoc}
        """
        num_cpu = os.cpu_count()
        cpu_count = int(num_cpu * 0.8) or 1
        with Pool(cpu_count) as pool:
            pool.map(self.performAnalysis_multiprocessing, self.images)
