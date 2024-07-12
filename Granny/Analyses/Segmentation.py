"""
This module performs instance segmentation on user-provided image files.
The analysis is conducted as follows:
    1. retrieves the machine learning (instance segmentation) trained models
       from https://osf.io/. to the current directory 'Analyses/'. The machine
       learning models are uploaded manually and should be named in this
       convention: granny-v{granny_version}-{model_name}-v{model_version}.pt
    2. parses user's input for image folder, initiates a list of Granny.Models.
       Images.Image, then runs YOLOv8 on the images.
    3. extracts and processes the segmented instances from the images.

date: July 12, 2024
author: Nhan H. Nguyen
"""

import colorsys
import os
import pathlib
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib import request

import cv2
import numpy as np
import pandas as pd
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


class SegmentationConfig:
    """
    Configuration class for instance segmentation module.

    This class provides the configuration details for different segmentation models,
    including the class labels and model details.

    Attributes:
        CLASSES (Dict[str, int]): A dictionary mapping class names to their respective integer labels.
            Format:
                {
                    "class_name_1": int_label_1,
                    "class_name_2": int_label_2,
                    ...
                }

        MODELS (Dict[str, Dict[str, str]]): A dictionary containing model details. Each model is represented
            by another dictionary that includes the full name of the model file and the URL to download it from.
            format:
                {
                    "model_name_1": {
                        "model_full_name_1": "*.pt",
                        "osf_urf_1": "https://osf_link/download/"
                    }
                }
                {
                    "model_name_2": {
                        "model_full_name_2": "*.pt",
                        "osf_urf_1": "https://osf_link/download/"
                    }
                }

    """

    CLASSES: Dict[str, int] = {"fruits": 0, "tray_info": 1}
    MODELS: Dict[str, Dict[str, str]] = {
        "pome_fruit-v1_0": {
            "full_name": "granny-v1_0-pome_fruit-v1_0.pt",
            "url": "https://osf.io/vyfhm/download/",
        }
    }


class Segmentation(Analysis):
    __analysis_name__ = "segmentation"

    def __init__(self):
        super().__init__()
        self.config = SegmentationConfig
        self.analysis_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

        # selects a model from the list to be used in this analysis
        self.models = self.config.MODELS
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
        self.seg_images = ImageListValue(
            "seg_img",
            "segmented_images",
            "The output directory where analysis' images are written.",
        )
        self.seg_images.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "segmented_images",
            )
        )
        self.tray_infos = ImageListValue(
            "info",
            "tray_info",
            "The output directory where tray information of the analysis' images are written.",
        )
        self.tray_infos.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "tray_infos",
            )
        )
        self.masked_images = ImageListValue(
            "f_img",
            "full_masked_image",
            "The output directory where the full-masked images are written.",
        )
        self.masked_images.setValue(
            os.path.join(
                os.curdir,
                "results",
                self.__analysis_name__,
                self.analysis_time,
                "full_masked_images",
            )
        )

        self.addInParam(self.model, self.input_images)

    def _getModelUrl(self, model_name: str):
        """
        Parses the self.models attribute to retrieves segmentation ML model URl using model_name.
        """
        model_url = ""
        try:
            model_url = self.models[model_name]["url"]
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

    def _segmentInstances(self, image: NDArray[np.uint8]) -> List[Any]:
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

    def _extractMaskedImage(self, tray_image: Image) -> Image:
        """
        Extracts a masked image from the given tray image, overlaying segmentation masks and bounding boxes.

        This method applies segmentation masks to the input tray image and draws bounding boxes around the detected objects.
        Each mask is colored uniquely, and the confidence score of each detected object is displayed on the image.

        Args:
            tray_image (Image): The input tray image from which to extract the masked image.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.

        Returns:
            Image: An image instance with the segmentation masks and bounding boxes overlaid.
        """
        [result] = tray_image.getSegmentationResults()
        masks = result.masks.cpu()
        boxes = result.boxes.cpu()
        coords = boxes.xyxy.cpu().numpy()
        confs = result.boxes.cpu().conf

        img = tray_image.getImage()
        result = img.copy()
        alpha = 0.5
        num_instances = masks.shape[0]
        brightness = 1.0
        hsv = [(i / num_instances, 1, brightness) for i in range(num_instances)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        for i in range(num_instances):
            mask = masks.data[i].numpy()
            (r, g, b) = colors[i]
            for c in range(3):
                result[:, :, c] = np.where(
                    mask == 1,
                    result[:, :, c] * (1 - alpha) + alpha * colors[i][c] * 255,
                    result[:, :, c],
                )

            x1, y1, x2, y2 = coords[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (r * 255, g * 255, b * 255), 5)
            cv2.putText(
                result,
                "{:.3f}".format(confs[i]),
                (x1, y1),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(255, 255, 255),
                thickness=3,
            )
        image_instance: Image = RGBImage(
            pathlib.Path(tray_image.getImageName()).stem + f"_masked_image" + ".png"
        )
        image_instance.setImage(result)
        return image_instance

    def _sortInstances(self, boxes: NDArray[np.float32], img_shape: Tuple[int, int]):
        """
        Helper function to sort the fruit tray using their center coordinates.

        This sorting algorithm follows the numbering convention in demo/numbering_tray_convention.pdf.
        In an increasing order, it sorts by y-center coordinates and then by x-center coordinates.

        Args:
            boxes (NDArray[np.float32]): A NumPy array of shape (N, 6), where N is the number of bounding boxes.
                Each row represents a bounding box with the format [x1, y1, x2, y2, conf, cls].
            img_shape (Tuple[int, int]): A tuple representing the shape of the image (height, width).

        Returns:
            pd.DataFrame: A DataFrame containing the sorted bounding boxes with additional columns:
                - ycenter: The y-coordinate of the center of the bounding box.
                - xcenter: The x-coordinate of the center of the bounding box.
                - rows: The row number assigned based on the y-center coordinates.
                - apple_id: The unique identifier assigned to each bounding box after sorting.
        """
        h, _ = img_shape
        df = pd.DataFrame(boxes)
        df.columns = ["x1", "y1", "x2", "y2", "conf", "cls"]
        df["ycenter"] = ((df["y1"] + df["y2"]) / 2).astype(int)
        df["xcenter"] = ((df["x1"] + df["x2"]) / 2).astype(int)
        df["rows"] = 0
        df["apple_id"] = 0
        df["nums"] = df.index
        df = df.sort_values("ycenter", ascending=True).reset_index(drop=True)
        df["rows"] = (df["ycenter"].diff().abs().gt(h // 20).cumsum() + 1).fillna(1).astype(int)

        df_list: List[pd.DataFrame] = []
        apple_id = 1
        increment = 1
        for i in range(1, df["rows"].max() + 1):
            dfx = (
                df[df["rows"] == i].sort_values("xcenter", ascending=False).reset_index(drop=True)
            )
            dfx["apple_id"] = range(apple_id, apple_id + increment * len(dfx), increment)
            df_list.append(dfx)
            apple_id += increment * len(dfx)

        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df

    def _extractTrayInfo(self, tray_image: Image) -> List[Image]:
        """
        Extracts individual tray information instances from the given tray image.

        This method identifies and isolates instances of the "tray_info" class within the provided tray image.
        It uses the segmentation results to extract bounding boxes and masks for the relevant class,
        sorts these instances, and then extracts and saves individual images for each detected instance.

        Args:
            tray_image (Image): The input tray image from which to extract tray information instances.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.

        Returns:
            List[Image]: A list of `Image` objects representing the individual tray information instances.
                If no instances of the "tray_info" class are found, an empty list is returned.
        """
        info_cls = SegmentationConfig.CLASSES["tray_info"]

        # gets bounding boxes, binary masks, and the original full-tray image array
        [results] = tray_image.getSegmentationResults()
        boxes = results.boxes.cpu().numpy()
        masks = results.masks.cpu().numpy()
        # checks for tray info class in segmentation results
        if not info_cls in boxes.cls:
            return []
        tray_idx: NDArray[int] = np.where(boxes.cls == info_cls)  # type:ignore
        boxes = boxes[tray_idx]
        masks = masks[tray_idx]
        tray_image_array: NDArray[np.uint8] = tray_image.getImage()

        # sorts boxes and masks based on xy-coordinates
        sorted_df = self._sortInstances(boxes.data, boxes.orig_shape)
        order: NDArray[np.float32] = sorted_df["nums"].to_numpy()
        sorted_boxes = boxes.data[order]  # type: ignore
        sorted_masks = masks.data[order]  # type: ignore

        # extracts instances based on bounding boxes and masks
        tray_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_name = (
                pathlib.Path(tray_image.getImageName()).stem + f"_tray_info_{i+1}" + ".png"
            )
            image_instance: Image = RGBImage(image_name)
            image_instance.setImage(individual_image)
            tray_images.append(image_instance)

        # returns a list of individual instances
        return tray_images

    def _extractImage(self, tray_image: Image) -> List[Image]:
        """
        Extracts individual fruit instances from the given tray image using binary masks.

        This method performs instance segmentation on the input tray image to isolate and extract
        images of each fruit detected by the YOLO model. It utilizes the segmentation results to
        obtain bounding boxes and masks, sorts the instances, and then extracts and saves individual
        images for each detected fruit.

        Args:
            tray_image (Image): The input tray image from which to extract fruit instances.
                This object is expected to have methods `getSegmentationResults()` and `getImage()`.

        Returns:
            List[Image]: A list of `Image` objects representing the individual fruit instances.
                If no instances of the "fruits" class are found, an empty list is returned.
        """
        fruit_cls = SegmentationConfig.CLASSES["fruits"]

        # gets bounding boxes, binary masks, and the original full-tray image array
        [results] = tray_image.getSegmentationResults()
        boxes = results.boxes.cpu().numpy()
        masks = results.masks.cpu().numpy()
        # checks for fruit class in segmentation results
        if not fruit_cls in boxes.cls:
            return []
        fruit_idx = np.where(boxes.cls == fruit_cls)  # type: ignore
        boxes = boxes[fruit_idx]
        masks = masks[fruit_idx]
        tray_image_array: NDArray[np.uint8] = tray_image.getImage()

        # sorts boxes and masks based on xy-coordinates
        sorted_df = self._sortInstances(boxes.data, boxes.orig_shape)
        order: NDArray[np.float32] = sorted_df["nums"].to_numpy()
        sorted_boxes = boxes.data[order]  # type: ignore
        sorted_masks = masks.data[order]  # type: ignore

        # extracts instances based on bounding boxes and masks
        individual_images: List[Image] = []
        for i in range(len(sorted_masks)):
            x1, y1, x2, y2, _, _ = sorted_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            individual_image = np.zeros([y2 - y1, x2 - x1, 3], dtype=np.uint8)
            mask = sorted_masks[i]
            for channel in range(3):
                individual_image[:, :, channel] = tray_image_array[y1:y2, x1:x2, channel] * mask[y1:y2, x1:x2]  # type: ignore
            image_name = pathlib.Path(tray_image.getImageName()).stem + f"_fruit_{i+1}" + ".png"
            image_instance: Image = RGBImage(image_name)
            image_instance.setImage(individual_image)
            individual_images.append(image_instance)

        # returns a list of individual instances
        return individual_images

    def performAnalysis(self) -> List[Image]:
        """
        {@inheritdoc}
        """
        self.model_name: str = self.in_params.get(self.model.getName()).getValue()  # type:ignore

        # download trained ML models from https://osf.io to the current directory
        if self.model_name.endswith(".pt"):
            self.local_model_path = self.model_name
        else:
            self.local_model_path = os.path.join(
                f"{pathlib.Path(__file__).parent}", self.models[self.model_name]["full_name"]
            )
            if not os.path.exists(self.local_model_path):
                model_url: str = self._getModelUrl(self.model_name)  # type: ignore
                self._downloadTrainedWeights(model_url)

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
        segmented_images: List[Image] = []
        tray_images: List[Image] = []
        masked_images: List[Image] = []
        for image_instance in self.images:
            # set ImageIO with specific file path
            self.image_io.setFilePath(image_instance.getFilePath())

            # loads image from file system with RGBImageFile(ImageIO)
            image_instance.loadImage(image_io=self.image_io)

            # rotates the image to landscape if the orientation is portrait
            h, w, _ = image_instance.getShape()
            if h > w:
                image_instance.rotateImage()

            # predicts fruit instances in the image
            result = self._segmentInstances(image=image_instance.getImage())

            # sets segmentation result
            image_instance.setSegmentationResults(results=result)

            try:
                # extracts individual instances
                image_instances = self._extractImage(image_instance)
                # and tray information
                tray_infos = self._extractTrayInfo(image_instance)
                # and masked image
                masked_image = self._extractMaskedImage(image_instance)

                # save to list for output
                segmented_images.extend(image_instances)
                tray_images.extend(tray_infos)
                masked_images.append(masked_image)
            except:
                AttributeError("Error with the results.")

        # 1. sets the output ImageListValue with the list of segmented images
        # 2. writes the segmented images to "segmented_images" folder
        # 3. writes the tray information to "tray_info" folder
        # 4. writes the full masked images to "full_masked_images" folder
        self.seg_images.setImageList(segmented_images)
        self.seg_images.writeValue()

        self.tray_infos.setImageList(tray_images)
        self.tray_infos.writeValue()

        self.masked_images.setImageList(masked_images)
        self.masked_images.writeValue()
        return segmented_images
