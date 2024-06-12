
from typing import Any, List
from urllib import request

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Analyses.Parameter import StringParam
from Granny.Models.AIModel.AIModel import AIModel
from Granny.Models.AIModel.YoloModel import YoloModel
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from numpy.typing import NDArray


class ImageAnalysis(Analysis):
    __analysis_name__ = "image_analysis"

    def __init__(self):
        """
        Initializes an instance of an Analysis object

        @param GRANNY.Models.Images.Image An instance of an Image object

        @return GRANNY.Analyses.Analysis.Analysis object.
        """
        super.__init__(self)
        self.images: List[Image] = []
        
    def addParams(self):
        """
        Adds the default parameters for all analyses.
        """
        self.input_dir = StringParam(
            "in", 
            "input", 
            "Input folder containing image files for the analysis."
        )
        self.output_dir = StringParam(
            "out", 
            "output", 
            "Output folder to export the analysis's results."
        )
        self.output_dir.setValue(
            os.path.join(Path(__file__).parent.parent.as_posix(), "results/")
        )  # Granny/results/

        self.addParam(self.input_dir, self.output_dir)