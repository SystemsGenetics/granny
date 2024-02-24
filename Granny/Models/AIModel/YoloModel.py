import colorsys
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.Images.Image import Image
from matplotlib import patches
from ultralytics import YOLO


class YoloModel(AIModel):
    def __init__(self, model_dir: str):
        AIModel.__init__(self, model_dir)

    def loadModel(self):
        """
        {@inheritdoc}
        """
        self.model = YOLO(self.model_dir)


    # todo: move this to segmentedimage
    def visualizeResults(self, image_instance: Image):
        """
        Draws boxes and overlays masks onto the image

        @param image_name: numpy array of the image.
        """
        # gets images and result from Image class
        image = image_instance.getImage()
        result = image_instance.getSegmentationResults()
        masks = result.masks # type: ignore
        boxes = result.boxes # type: ignore
        xyxy = boxes.xyxy.numpy().astype(int) # type: ignore

        # generates random colors.
        # to get visually distinct colors, generate them in HSV space then
        # convert to RGB.
        alpha = 0.5
        num_instances = result.__len__()
        brightness = 1.0
        hsv = [(i / num_instances, 1, brightness) for i in range(num_instances)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)

        # applies masks and boxes to the image
        _, ax = plt.subplots()
        for i in range(num_instances):
            mask = masks.data[i].numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            for c in range(3):
                image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * colors[i][c]*255, image[:, :, c])
            x1, y1, x2, y2 = xyxy[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    edgecolor=colors[i],
                                    facecolor='none', linestyle="dashed")
            ax.add_patch(p)

        # displays the image and saves the overlayed image to file
        plt.axis("off")
        plt.imshow(image)
        plt.tight_layout()
        plt.savefig("", bbox_inches='tight', pad_inches=0)
        plt.show()
