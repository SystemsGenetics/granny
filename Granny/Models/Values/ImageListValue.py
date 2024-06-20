import os
from typing import List

from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage
from Granny.Models.IO.ImageIO import ImageIO
from Granny.Models.IO.RGBImageFile import RGBImageFile
from Granny.Models.Values.FileDirValue import FileDirValue


class ImageListValue(FileDirValue):
    """
    A value that stores the directory path where images are kept.

    The readValue() and writeValue() functions will read and write
    the images that are in the directory provided as the value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value: str = ""
        self.images: List[Image] = []

        self.IMAGE_EXTENSION = (
            ".JPG",
            ".JPG".lower(),
            ".PNG",
            ".PNG".lower(),
            ".JPEG",
            ".JPEG".lower(),
            ".TIFF",
            ".TIFF".lower(),
        )

    def readValue(self):
        """
        This method reads the string value as stored in self.value and returns a list of
        Granny.Models.Image.Image objects
        """
        # reads image files from the input directory
        image_files: List[str] = os.listdir(self.value)  # type: ignore
        images = []
        for image_file in image_files:
            if image_file.endswith(self.IMAGE_EXTENSION):
                rgb_image = RGBImage(os.path.join(self.value, image_file))
                images.append(rgb_image)
        self.images = images

    def writeValue(self):
        """ """
        image_io: ImageIO = RGBImageFile()
        for image in self.images:
            image.saveImage(image_io)
        # @todo write the metadata for the image that
        # lives along side of the image.

    def getImageList(self):
        """ """
        return self.images

    def setImageList(self, images: List[Image]):
        """ """
        self.images = images

    def getDefaultValue(self) -> str:
        """
        {@inheritdoc}
        """
        return self.default_value

    def setDefaultValue(self, value: str):
        """
        {@inheritdoc}
        """
        self.default_value = value
