"""
This class perform instance segmentation on the user provided image files. The analysis will be
carried out in the following manner:
    1. retrieves the machine learning (instance segmentation) trained models from https://osf.io/.
    to the current directory 'Analyses/'. The machine learning models are uploaded manually and should be
    named in this convention: granny-v{granny_version}-{model_name}-v{model_version}.pt
    2. parses user's input for image folder, initiates a list of Granny.Models.Images.Image,
    then runs YOLOv8 on the images.
    3. Todo: visualize results and segment individual images

date: June 06, 2024
author: Nhan H. Nguyen
"""

import configparser
import os
import pathlib
from typing import Any, List
from urllib import request

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray

# I think we should keep the config_path but the software should ask for the user's input for the
# model name in self.model_name
CONFIG_PATH = "config/granny-v1_0/segmentation.ini"


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self):
        Analysis.__init__(self)

        # name of the model to be used in this analysis
        self.model_name = "granny-v1_0-pome_fruit-v1_0.pt"

        # download trained ML models from https://osf.io to the current directory
        self.local_model_path = os.path.join(f"{pathlib.Path(__file__).parent}", self.model_name)
        self.model_url = self.getModelUrl(self.model_name)
        print(self.model_url)
        if not os.path.exists(self.local_model_path):
            self.downloadTrainedWeights(self.local_model_path)

        # loads segmentation model
        self.AIModel: AIModel = YoloModel(self.local_model_path)
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

    def getModelUrl(self, model_name: str) -> str:
        """ """
        config = configparser.ConfigParser()
        config.read(os.path.join(f"{pathlib.Path(__file__).parent}", CONFIG_PATH))
        return config["Models"][model_name]

    def downloadTrainedWeights(self, local_model_path: str, verbose: int = 1):
        """Download YOLO8 trained weights from Granny GitHub repository:
        https://github.com/SystemsGenetics/granny/tree/dev-MVC/Granny/Analyses/config/segmentation/

        @param local_model_path: local path of the trained weights
        """
        if verbose > 0:
            print(f"Downloading pretrained model to {local_model_path} ...")
        request.urlretrieve(self.model_url, local_model_path)
        if verbose > 0:
            print("... done downloading pretrained model!")

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
        # initiates Granny.Model.Images.Image instances for the analysis
        self.images = self.getImages()

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
