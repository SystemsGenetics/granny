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
from Granny.Analyses.ImageAnalysis import ImageAnalysis
from Granny.Analyses.Parameter import ImageListParam
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray

class Segmentation(ImageAnalysis):
    __analysis_name__ = "segmentation"    

    def __init__(self, images: List[Image], th: int):
        super.__init__(self)

        self.models = {
            'pome_fruit-v1_0': {
                'url': 'https://osf.io/dqzyn/download',
                'path': './granny-v1_0-pome_fruit-v1_0.pt'
            }
        }

        # name of the model to be used in this analysis (this could be changed to take user's input)
        self.model_name = "granny-v1_0-pome_fruit-v1_0.pt"

        # download trained ML models from https://osf.io to the current directory
        self.local_model_path = os.path.join(f"{pathlib.Path(__file__).parent}", self.model_name)

        if not os.path.exists(self.local_model_path):
            self.model_url = self.getModelUrl(self.model_name)
            self.downloadTrainedWeights(self.model_url)

        # loads segmentation model
        self.AIModel: AIModel = YoloModel(self.local_model_path)
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

        self.addInParams()

    def addInParams(self):
        """
        Adds all of the parameters that the Segmentation analysis needs.
        """
        model_param = ImageListParam(
            "model", "model", "Specifies the model that should be used for segmentation. The " +
              "model can be specified in one of three ways. First, if a known model name is provided " +
              "(e.g. 'pome_fruit-v1_0') then Granny will automatically retrieve the model.  If " +
              "a URL is provided then Granny will download the model from the URL you provided. " +
              "Otherwise the value must be a path to where the model is stored on the local file system."
        )
        self.addInParam(model_param)

    def addOutParams(self):
        """
        Adds all of the parameters that the Segmentation analysis needs.
        """
        masked_image = ImageListParam(
            "masked_image", 
            "masked_image", 
            "The list of images after segementation."
        )
        self.addOutParam(masked_image)

        segmented_images = ImageListParam(
            "segmented_images", 
            "segmented_images", 
            "The list of images after segmentation."
        )
        self.addOutParam(segmented_images)

    def getModelUrl(self, model_name: str):
        """
        Parses the config file 'config/granny-v1_0/segmentation.ini' to retrieves
        segmentation ML model URl using model_name as key.
        """
        config = configparser.ConfigParser()
        config.read(os.path.join(f"{pathlib.Path(__file__).parent}", CONFIG_PATH))
        model_url = ""
        try:
            model_url = config["Models"][model_name]
            print(f"Model URL: {model_url}")
        except KeyError:
            print(f"Key '{model_name}' not found in configuration.")
        return model_url

    def downloadTrainedWeights(self, model_url: str, verbose: int = 1):
        """Download YOLO8 trained weights from Granny GitHub repository:
        https://github.com/SystemsGenetics/granny/tree/dev-MVC/Granny/Analyses/config/segmentation/

        @param local_model_path: local path of the trained weights
        """
        if verbose > 0:
            print(f"Downloading pretrained model to {self.local_model_path} ...")
        request.urlretrieve(model_url, self.local_model_path)
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

    def extractFeature(self, tray_image: Image) -> List[Image]:
        """
        From the given full 'tray_image', using the binary masks stored in 'results', performs
        instance segmentation to extract each YOLO-detected feature.
        """
        # gets bounding boxes, binary masks, and the original full-tray image array
        [results] = tray_image.getSegmentationResults()
        boxes: NDArray[np.float32] = results.boxes.data.numpy()  # type: ignore
        masks: NDArray[np.float32] = results.masks.data.numpy()  # type: ignore
        tray_image_array: NDArray[np.uint8] = tray_image.getImage()

        # sorts boxes and masks based on y-coordinates
        # todo: sort them using both x and y coordinates as numbering convention
        y_order = boxes[:, 1].argsort()  # type: ignore
        sorted_boxes = boxes[y_order]  # type: ignore
        sorted_masks = masks[y_order]  # type: ignore

        # extracts instances based on bounding boxes and masks
        individual_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_instance: Image = RGBImage(tray_image.getImageName())
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # returns a list of individual instances
        return individual_images

    def performAnalysis(self) -> List[Image]:
        """
        {@inheritdoc}
        """
        # initiates user's input
        # self.input_dir: StringParam = self.params.get(self.input_dir.getName())  # type:ignore
        # self.output_dir: StringParam = self.params.get(self.output_dir.getName())  # type:ignore

        # initiates Granny.Model.Images.Image instances for the analysis
        self.images = self.params.get('input').getValue()

        # initiates ImageIO
        self.image_io: ImageIO = RGBImageFile()

        # performs segmentation on each image one-by-one
        image_instances: List[Image] = []
        for image in self.images:
            # set ImageIO with specific file path
            self.image_io.setFilePath(image.getFilePath())

            # loads image from file system with RGBImageFile(ImageIO)
            image.loadImage(image_io=self.image_io)

            # predicts fruit instances in the image
            result = self.segmentInstances(image.getImage())

            # sets segmentation result
            image.setSegmentationResults(results=result)
            image_instances.append(image)

        # replaces the image list with the images containing the segmentation result
        self.images = image_instances

        image_instances = self.extractFeature(self.images[0])
