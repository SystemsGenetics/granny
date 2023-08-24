import os
import pathlib
import sys

sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "Mask_RCNN"))
from Mask_RCNN import config as MRCNN_config
from Mask_RCNN import model as MRCNN_model
from Mask_RCNN import utils as MRCNN_utils
from Mask_RCNN import visualize as MRCNN_visualize


class AppleConfig(MRCNN_config.Config):
    NUM_CLASSES = 1 + 1
    NAME = "GS_Apple"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    BATCH_SIZE = 1
