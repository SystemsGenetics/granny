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
    """
    {@inheritdoc}
    """

    def __init__(self, model_dir: str):
        AIModel.__init__(self, model_dir)

    def loadModel(self):
        """
        {@inheritdoc}
        """
        self.model = YOLO(self.model_dir)

    def getModel(self):
        """
        {@inheritdoc}
        """
        return self.model
