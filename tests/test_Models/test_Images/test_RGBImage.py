from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.StringValue import StringValue
# Sets and gets the metadata from the object.
def test_getValue_getFilePath():
    filePath = "/test-assets/images/single"
    image1 = RGBImage(filePath)
    image1.metadata = {"test": 1}
    assert image1.getValue("test") == 1
    assert image1.getFilePath() == filePath

# Sets and tests if the image was properly set by comparing it to the same image
# by directly setting the value.
def test_get_setImage():
    filePath = "/test-assets/images/single"
    image1 = RGBImage(filePath)
    imageTest = RGBImage(filePath)
    imageTest.image = ("/test-assets/images/single/cross_section_demo_image.jpeg")
    image1.setImage("/test-assets/images/single/cross_section_demo_image.jpeg")
    assert image1.getImage() == imageTest.getImage()



# Creates a RGBImage and RGBImageFile object and sets the imageIO object
# to the image of intrest then sets the image object to that value.
# It is then rotated 90 degrees clockwise and outputed into the specific folder.
# TODO check the if the image properly rotated within the code without
# looking at the image directly. 
def test_save_load_rotateImage():
    filePath = "/test-assets/images/single"
    image1 = RGBImage(filePath)
    imageActual = RGBImageFile()
    imageActual.setFilePath("test-assets/images/single/test_RGBImage_rotate.jpeg")
    imageTest2 = imageActual.loadImage()
    image1.loadImage(imageActual)
    image1.rotateImage()
    image1.saveImage(imageActual,"test-assets/results")
    assert image1 
# Sets image1 to demo image, then converts it to BGR and then back to RBG
def test_toRBG_toBGR():
    filePath = "test-assets/images/single"
    image1 = RGBImage(filePath)
    imageActual = RGBImageFile()
    imageActual.setFilePath("test-assets/images/single/test_RGBImage_BGR_Test.jpeg")
    #imageTest2 = imageActual.loadImage()
    image1.loadImage(imageActual)
    image1.toBGR()
    image1.saveImage(imageActual,"test-assets/results")
    imageActual.setFilePath("test-assets/images/single/test_RGBImage_RGB_Test.jpeg")
    image1.toRGB()
    image1.saveImage(imageActual,"test-assets/results")
    assert image1
# Checks if the MetaData properly outputs the correct version of the dictionary
def test_get_setMetaData():
    a = StringValue("name1", "label1", "help1")
    b = StringValue("name2", "label2", "help2")
    c = StringValue("name3", "label3", "help3")

    mData = {1:a,2:b,3:c}
    filePath = "/test-assets/images/single"
    image1 = RGBImage(filePath)
    image1.setMetaData(mData)
    assert image1.getMetaData() == {"name1": a, "name2": b, "name3": c}
# Sets the metadata results to a string then sees if the string is returned
def test_set_get_checkResults():
    filePath = "/test-assets/images/single"
    image1 = RGBImage(filePath)
    image1.setSegmentationResults("Results")
    assert image1.getSegmentationResults() == "Results"
    assert not image1.checkResult()