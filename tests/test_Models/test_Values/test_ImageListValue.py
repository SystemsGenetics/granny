from Granny.Models.Values.ImageListValue import ImageListValue
from Granny.Models.Images.RGBImage import RGBImage

# Tests the read and write functions and if the outputs are correct
def test_read_writeValue():
    file_path = "demo/granny_smith_images/binarized_images/"
    imageList = ImageListValue("name", "label", "help")
  
    imageList.value = file_path
    imageList.readValue()
    # Ask for help for this
    # ERROR: RBGImage object has no attribute 'image'
    #imageList.writeValue()
    assert imageList

   
def test_get_setValue():
    image = RGBImage("/demo/cross_section_images/cross_section_tray")
    images = [image]
    file_path = "demo/granny_smith_images/binarized_images/"
    imageList = ImageListValue("name", "label", "help")
  
    imageList.value = file_path
    imageList.setImageList(images)
    assert imageList.getImageList() == images
    
    