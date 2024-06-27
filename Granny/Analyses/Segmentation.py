"""
This class perform instance segmentation on the user provided image files.
The analysis will be carried out in the following manner:
    1. retrieves the machine learning (instance segmentation) trained models
       from https://osf.io/. to the current directory 'Analyses/'. The machine
       learning models are uploaded manually and should be named in this
       convention: granny-v{granny_version}-{model_name}-v{model_version}.pt
    2. parses user's input for image folder, initiates a list of Granny.Models.
       Images.Image, then runs YOLOv8 on the images.
    3.

date: June 06, 2024
author: Nhan H. Nguyen
"""

import os
import pathlib
from typing import Any, List
from urllib import request

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.FileNameValue import FileNameValue
from Granny.Models.Values.ImageListValue import ImageListValue
from numpy.typing import NDArray


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self):
        super().__init__()

        # selects a model from the list to be used in this analysis
        self.models = {
            "pome_fruit-v1_0": {
                "full_name": "granny-v1_0-pome_fruit-v1_0.pt",
                "url": "https://osf.io/dqzyn/download/",
            }
        }
        self.model = FileNameValue(
            "model",
            "model",
            "Specifies the model that should be used for segmentation to identify fruit. The "
            + "model can be specified using a known model name (e.g. 'pome_fruit-v1_0'), "
            + "and Granny will automatically retrieve the model from the online "
            + "https://osf.io. "
            + "Otherwise the value must be a path to where the model is stored on the local "
            + "file system. If no model is specified then the default model will be used.",
        )        
        self.model.setValue("pome_fruit-v1_0")
        self.model.setIsRequired(False)

        self.input_images: ImageListValue = ImageListValue(
            "input", "input", "The directory where input images are located."
        )
        self.input_images.setIsRequired(True)

        self.output_images = ImageListValue(
            "output", "output", "The output directory where analysis' images are written."
        )
        self.output_images.setValue(os.path.join(os.curdir, self.__analysis_name__, "results"))
        self.addInParam(self.model, self.input_images)

    def _getModelUrl(self, model_name: str):
        """
        Parses the self.models attribute to retrieves segmentation ML model URl using model_name.
        """
        model_url = ""
        try:
            model_url = self.models[model_name]["url"]
            print(f"Model URL: {model_url}")
        except KeyError:
            print(f"Key '{model_name}' not found in configuration.")
        return model_url

    def _downloadTrainedWeights(self, model_url: str, verbose: int = 1):
        """
        Download YOLO8 trained weights from Granny GitHub repository:
        https://github.com/SystemsGenetics/granny/tree/dev-MVC/Granny/Analyses/config/segmentation/

        @param local_model_path: local path of the trained weights
        """
        if verbose > 0:
            print(f"Downloading pretrained model to {self.local_model_path} ...")
        request.urlretrieve(model_url, self.local_model_path)
        if verbose > 0:
            print("... done downloading pretrained model!")

    def _segmentInstances(self, image: NDArray[np.uint8], device: str = "cpu") -> List[Any]:
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
        results = self.segmentation_model.predict(image, device=device, retina_masks=True)  # type: ignore

        return results

    def _extractFeature(self, tray_image: Image) -> List[Image]:
        """
        From the given full 'tray_image', using the binary masks stored in 'results', performs
        instance segmentation to extract each YOLO-detected feature.
        """
        # gets bounding boxes, binary masks, and the original full-tray image array
        [results] = tray_image.getSegmentationResults()
        boxes: NDArray[np.float32] = results.boxes.data.cpu().numpy()  # type: ignore
        masks: NDArray[np.float32] = results.masks.data.cpu().numpy()  # type: ignore
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
            image_name = pathlib.Path(tray_image.getImageName()).stem + str(i) + ".png"
            image_instance: Image = RGBImage(image_name)
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # returns a list of individual instances
        return individual_images

    def performAnalysis(self) -> List[Image]:
        """
        {@inheritdoc}
        """
        self.model_name = self.in_params.get(self.model.getName()).getValue()  # type:ignore

        # download trained ML models from https://osf.io to the current directory
        self.local_model_path = os.path.join(
            f"{pathlib.Path(__file__).parent}", self.models[self.model_name]["full_name"]
        )

        if not os.path.exists(self.local_model_path):
            self.model_url: str = self._getModelUrl(self.model_name)  # type: ignore
            self._downloadTrainedWeights(self.model_url)

        # loads segmentation model
        self.AIModel: AIModel = YoloModel(self.local_model_path)
        self.AIModel.loadModel()
        self.segmentation_model = self.AIModel.getModel()

        # initiates user's input
        self.input_images = self.in_params.get(self.input_images.getName())  # type:ignore

        # initiates Granny.Model.Images.Image instances for the analysis using the user's input
        self.input_images.readValue()
        self.images = self.input_images.getImageList()

        # initiates ImageIO
        self.image_io: ImageIO = RGBImageFile()

        # performs segmentation on each image one-by-one
        output_images: List[Image] = []
        for image in self.images:
            # set ImageIO with specific file path
            self.image_io.setFilePath(image.getFilePath())

            # loads image from file system with RGBImageFile(ImageIO)
            image.loadImage(image_io=self.image_io)

            # predicts fruit instances in the image
            result = self._segmentInstances(image.getImage())

            # sets segmentation result
            image.setSegmentationResults(results=result)
            output_images.append(image)

        output_image_list: List[Image] = []
        # gets individual (segmented) images from mask and full-size image
        for output_image in output_images:
            image_instances = self._extractFeature(output_image)
            output_image_list.extend(image_instances)

        # 1. sets the output ImageListValue with the list of segmented images
        # 2. writes the segmented images to a folder
        self.output_images.setImageList(output_image_list)
        self.output_images.writeValue()
        self.addRetValue(self.output_images)

        return output_image_list
