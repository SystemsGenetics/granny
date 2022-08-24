import sys 
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.resolve(), "Mask_RCNN"))
import config as MRCNN_config
import utils as MRCNN_utils
import model as MRCNN_model
import visualize as MRCNN_visualize

class AppleConfig(MRCNN_config.Config):
    NUM_CLASSES = 1 + 1
    NAME = "GS_Apple"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    BATCH_SIZE = 1

