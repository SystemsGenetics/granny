import os

from Granny.Models.Values.FileNameValue import FileNameValue


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
            os.path.join(f"{self.output_dir.getValue()}", self.__analysis_name__, "ratings.csv"),
            "w",
        ) as file:
            sep = ","
            for image_instance in image_instances:
                output = ""
                for param in image_instance.getMetaData().getValueeters():
                    output = output + str(param.getValue()) + sep
                file.writelines(f"{image_instance.getImageName()}{sep}{output}")
                file.writelines("\n")

    def setDefaultValue(self, value: str):
        return super().setDefaultValue(value)

    def getDefaultValue(self) -> str:
        return super().getDefaultValue()
