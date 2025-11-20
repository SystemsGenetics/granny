from Granny.Models.IO.RGBImageFile import RGBImageFile
import uuid

# Sets the image to one of the demo images and checks if it properly saves
# in the right dir.
def test_load_saveImage_getType():
    image = RGBImageFile()
    image.setFilePath("test-assets/images/single/cross_section_demo_image.jpeg")
    test = image.loadImage()
    image.saveImage(test, "test-results/")
    assert image.getType() == "rgb" 