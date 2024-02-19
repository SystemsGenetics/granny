import os
import pathlib
import re
from typing import List, Tuple


class GrannyBase(object):
    """
    Base class for Granny - a Computer Vision software to perform fruit
    assessment from image.
    Granny's subclasses includes:
        - GRANNY_Segmentation: implementation of Mask-RCNN - an instance
        segmentation machine learning model, to extract instances from image
        input.
        - GRANNY_SuperficialScald: image processing to threshold superficial
        scald in "Granny Smith" apples.
        - GRANNY_PeelColor: image processing to extract green-yellow mean
        values on apple's/pear's peel.
        - and more ...
    """

    def __init__(
        self,
        action: str = "",
        fname: str = "",
    ):
        # current directory
        self.ROOT_DIR = pathlib.Path(__file__).parent.resolve()

        # logs
        self.MODEL_DIR = os.path.join(os.path.curdir, "logs")

        # new directory of the rotated input images
        self.NEW_DATA_DIR = "input_data" + os.sep

        # directory to download the pretrained model
        self.PRETRAINED_MODEL = os.path.join(
            self.ROOT_DIR, "mask_rcnn_starch_cross_section.h5"
        )

        # accepted file extensions
        self.IMAGE_EXTENSION = (
            ".JPG",
            ".JPG".lower(),
            ".PNG",
            ".PNG".lower(),
            ".JPEG",
            ".JPEG".lower(),
            ".TIFF",
            ".TIFF".lower(),
        )

        # initialize default parameters
        self.ACTION = action
        self.FILE_NAME = fname if fname.endswith(self.IMAGE_EXTENSION) else ""
        self.FOLDER_NAME = (
            fname
            if not fname.endswith(self.IMAGE_EXTENSION)
            else os.sep.join(fname.split(os.sep)[0:-1])
        )
        self.FOLDER_NAME = (
            pathlib.Path(self.FOLDER_NAME).as_posix()
            if not self.FOLDER_NAME.endswith(os.sep)
            else self.FOLDER_NAME
        )
        self.OLD_DATA_DIR = self.FOLDER_NAME
        self.INPUT_FNAME = fname
        self.RESULT_DIR = pathlib.Path("results").as_posix()

        # location where masked apple trays will be saved
        self.FULLMASK_DIR = f"{self.RESULT_DIR}{os.sep}full_masked_images{os.sep}"

        # location where segmented/individual instances will be saved
        self.SEGMENTED_DIR = f"{self.RESULT_DIR}{os.sep}segmented_images{os.sep}"

        # location where apples with the scald removed will be saved
        self.BINARIZED_IMAGES = f"{self.RESULT_DIR}{os.sep}binarized_images"

        # location where blush-isolated pear images will be saved
        self.BLUSHED_IMAGES = f"{self.RESULT_DIR}{os.sep}blush_images"

        # results for pear color bining
        self.BIN_COLOR = f"{self.RESULT_DIR}{os.sep}peel_color_results"

        # results for cross-section starch area
        self.STARCH_RESULTS = f"{self.RESULT_DIR}{os.sep}starch{os.sep}results"

        # results for cross-section starch area labels
        self.STARCH_LABELS = f"{self.RESULT_DIR}{os.sep}starch{os.sep}labels"

        # results for cross-section starch area labels
        self.STARCH_IMAGES = f"{self.RESULT_DIR}{os.sep}starch{os.sep}images"

    def create_directories(self, *args: str) -> None:
        """
        Create directories from args
        """
        for directory in args:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def list_all(self, data_dir: str = os.path.curdir) -> Tuple[List[str], List[str]]:
        """
        Recursively list all the folder names and image file names in the
        directory
        """
        file_name: List[str] = []
        folder_name: List[str] = []

        # if data_dir is a file
        if data_dir.endswith(self.IMAGE_EXTENSION):
            file_name.append(data_dir)
            folder_name.append(data_dir.replace(data_dir.split(os.sep)[-1], ""))
            return folder_name, file_name

        # list all folders and files in data_dir
        for root, dirs, files in os.walk(data_dir):
            # append the files to the list
            for file in files:
                if file.endswith(self.IMAGE_EXTENSION):
                    file_name.append(os.path.join(root, file))

            # append the folders to the list
            for fold in dirs:
                folder_name.append(os.path.join(root, fold))
            if folder_name == []:
                folder_name.append(os.path.join(root))

        return folder_name, file_name

    def clean_name(self, fname: str) -> str:
        """
        Remove image extensions in file names
        """
        for ext in self.IMAGE_EXTENSION:
            fname = re.sub(ext, "", fname)
        return fname
