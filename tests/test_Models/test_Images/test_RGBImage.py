from Granny.Models.Images.RGBImage import RGBImage

def test_getValue_getFilePath():
    filePath = "/demo/cross_section_images/cross_section_tray"
    image1 = RGBImage(filePath)
    image1.metadata = {"test": 1}
    assert image1.getValue("test") == 1
    assert image1.getFilePath() == filePath

def test_get_setImage():
    filePath = "/demo/cross_section_images/cross_section_tray"
    image1 = RGBImage(filePath)
    assert image1.getImage() == 1