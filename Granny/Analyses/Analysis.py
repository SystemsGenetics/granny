"""
Base abstract Analysis class for the analyses to be called by either the command line interface
or the graphical user interface.

Author: Nhan Nguyen
Date: May 21, 2024
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List

from Granny.Analyses.Parameter import Param, StringParam
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage

IMAGE_EXTENSION = (
    ".JPG",
    ".JPG".lower(),
    ".PNG",
    ".PNG".lower(),
    ".JPEG",
    ".JPEG".lower(),
    ".TIFF",
    ".TIFF".lower(),
)


class Analysis(ABC):

    __analysis_name__ = "analysis"

    def __init__(self):
        """
        Intializes an instance of an Analysis object

        @param GRANNY.Models.Images.Image An instance of an Image object

        @return GRANNY.Analyses.Analysis.Analysis object.
        """
        self.images: List[Image] = []
        self.params: List[Param] = []

        # initiates input and output directory
        self.input_dir = StringParam(
            "in", "input", "Input folder containing image files for the analysis."
        )
        self.output_dir = StringParam(
            "out", "output", "Output folder to export the analysis's results."
        )
        self.output_dir.setDefaultValue(
            os.path.join(Path(os.getcwd()).parent.parent.as_posix(), "results/")
        )  # Grannny/results/

        self.addParam(self.input_dir, self.output_dir)

    def setParam(self, params: List[Param]) -> None:
        """
        Sets the parameter list
        """
        self.params = params

    def addParam(self, *param: Param):
        """
        Adds a parameter to the parameter list
        """
        for p in param:
            self.params.append(p)

    def getParams(self) -> List[Param]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return list(self.params)

    def getImages(self) -> List[Image]:
        """
        Returns to the user the image list
        """
        input_dir: str = self.input_dir.getValue()
        image_files: List[str] = os.listdir(input_dir)
        for image_file in image_files:
            if image_file.endswith(IMAGE_EXTENSION):
                rgb_image = RGBImage(os.path.join(input_dir, image_file))
                self.images.append(rgb_image)
        return list(self.images)

    def generateAnalysisMetadata(self):
        """
        Generates general metadata for the analysis, including: date and time, analysis name, id.
        """
        # the analysis date and time
        time = StringParam("dt", "datetime", "Date and time of when the analysis was performed.")
        time.setDefaultValue(datetime.now().strftime("%Y-%m-%d %H:%M"))

        # the analysis id
        id = StringParam("id", "identifier", "Unique identifier for the analysis")
        id.setDefaultValue(str(uuid.uuid4()))

        self.addParam(time, id)

    @abstractmethod
    def performAnalysis(self) -> None:
        """
        Once all required paramterers have been set, this function is used
        to perform the analysis.
        """
        pass
