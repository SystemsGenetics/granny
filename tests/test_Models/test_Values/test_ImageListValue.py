from Granny.Models.Values.ImageListValue import ImageListValue
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.RGBImageFile import RGBImageFile

# Tests the read and write functions and if the outputs are correct
def test_read_writeValue():
    file_path = "demo/granny_smith_images/binarized_images/"
    imageList = ImageListValue("name", "label", "help")
  
    imageList.value = file_path
    imageList.readValue()
    """
    The writeValue needs the image to be loaded first (converted into an np array)
    before it can be saved. I idea is to go through the list of images
    which are of RGBImage type and set an RGBImageFile object to the filepath
    of the image. With a temp object with the corresponding filepath set,
    the image is then loaded which converts the image into an np array. 
    the converted image is then set to the image of the imageList in which
    we are accessing. 
    """
    image_io = RGBImageFile()
    for i, image in enumerate(imageList.images):
        image_io.filepath = image.filepath
        imageList.images[i].image = image_io.loadImage()
    
    imageList.writeValue()
    assert imageList

# Tests the get and set value by creating a list of RBGImages and comparing
# the results with the list
def test_get_setValue():
    image = RGBImage("/demo/cross_section_images/cross_section_tray")
    images = [image]
    file_path = "demo/granny_smith_images/binarized_images/"
    imageList = ImageListValue("name", "label", "help")
  
    imageList.value = file_path
    imageList.setImageList(images)
    assert imageList.getImageList() == images
    
