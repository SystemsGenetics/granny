import os
import pathlib
import shutil
from typing import Any, List
from urllib import request

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

        self.local_model_path = os.path.join(
            f"{pathlib.Path(__file__).parent}",
            "config",
            self.__analysis_name__,
            "granny-v1_0-pome_fruit-v1_0.pt",
        )
        self.model_url = ""
        if not os.path.exists(self.local_model_path):
            os.makedirs(pathlib.Path(self.local_model_path).parent)
            self.downloadTrainedWeights(self.local_model_path)

        # loads segmentation model
        self.AIModel: AIModel = YoloModel(self.local_model_path)
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

    def downloadTrainedWeights(self, local_model_path: str, verbose: int = 1):
        """Download YOLO8 trained weights from Granny GitHub repository:
        https://github.com/SystemsGenetics/granny/tree/dev-MVC/Granny/Analyses/config/segmentation/

        @param local_model_path: local path of the trained weights
        """
        if verbose > 0:
            print(f"Downloading pretrained model to {local_model_path} ...")
        with request.urlopen(self.model_url) as resp, open(local_model_path, "wb") as out:
            shutil.copyfileobj(resp, out)
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
