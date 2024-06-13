import os
from pathlib import Path
from typing import Any, List
from urllib import request

import numpy as np
from Granny.Analyses.Analysis import Analysis
from Granny.Analyses.Values import StringValue
from Granny.Analyses.Values import ImageListValue
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
        super().__init__()
        self.images: List[Image] = []
       
        image_list = ImageListValue(
            "input", 
            "input", 
            "The directory where input images are located."
        )
        self.addParam(image_list)