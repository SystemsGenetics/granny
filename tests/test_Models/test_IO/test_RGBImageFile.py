from Granny.Models.IO.RGBImageFile import RGBImageFile
import uuid

# Sets the image to one of the demo images and checks if it properly saves
# in the right dir.
def test_load_saveImage_getType():
    image = RGBImageFile()
    image.setFilePath("demo/cross_section_images/cross_section_tray/cross_section_demo_image.jpeg")
    test = image.loadImage()
    image.saveImage(test, "tmp/")
    assert image.getType() == "rgb" 