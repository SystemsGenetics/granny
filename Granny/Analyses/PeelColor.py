from Granny.Analyses.Analysis import Analysis
from Granny.Models.Images.Image import Image


class PeelColor(Analysis):
    __analysis_name__ = "color"

    def __init__(self, image: Image):
        Analysis.__init__(self, image)
