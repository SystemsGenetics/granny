import os
import pathlib
import sys

from Mask_RCNN import config as MRCNN_config

sys.path.append(
    os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "Mask_RCNN")
)


class AppleConfig(MRCNN_config.Config):
    NUM_CLASSES = 1 + 1
    NAME = "GS_Apple"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    BATCH_SIZE = 1
