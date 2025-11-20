from Granny.Analyses.StarchArea import StarchArea
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.StringValue import StringValue
from Granny.Models.Values.ImageListValue import ImageListValue


def test_StarchAnalyses():
    analysis = StarchArea()
    # Set up input images from demo directory
    analysis.input_images.setValue("demo/cross_section_images/full_masked_images")

    # This test just verifies that StarchArea can be instantiated
    # and that input_images can be set without errors
    assert analysis.input_images.getValue() == "demo/cross_section_images/full_masked_images"