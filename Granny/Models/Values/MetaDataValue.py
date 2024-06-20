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
        pass
