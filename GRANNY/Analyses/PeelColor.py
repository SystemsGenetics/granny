from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Models.Images.Image import Image


class PeelColor(Analysis):
    def __init__(self, image: Image):
        Analysis.__init__(self, image)
