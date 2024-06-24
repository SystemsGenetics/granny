import os
from typing import List

from Granny.Models.Images.Image import Image
from Granny.Models.Values.FileNameValue import FileNameValue
from Granny.Models.Values.Value import Value


class MetaDataValue(FileNameValue):
    """
    A value that stores the file name where the metadata are kept.

    The readValue() and writeValue() functions will read and write
    the metadata into the filename of the value.
    """

    def __init__(self, name: str, label: str, help: str):
        super().__init__(name, label, help)

    def readValue(self):
        """ """
        pass

    def writeValue(self):
        """ """
        with open(
            os.path.join(f"{self.getValue()}"),
            "w",
        ) as file:
            sep = ","
            for image_instance in images:
                output = ""
                for param in image_instance.getMetaData().getValues():
                    output = output + str(param.getValue()) + sep
                file.writelines(f"{image_instance.getImageName()}{sep}{output}")
                file.writelines("\n")
