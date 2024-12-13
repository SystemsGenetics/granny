from Granny.Analyses.StarchArea import StarchArea
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.StringValue import StringValue
from Granny.Models.Values.ImageListValue import ImageListValue


def test_StarchAnalyses():
    analysis = StarchArea()
    #images = ImageListValue("test","test","test")
    #analysis.addInParam(images) 

    analysis.performAnalysis()