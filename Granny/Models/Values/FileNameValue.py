import os

from Granny.Models.Values.StringValue import StringValue


class FileNameValue(StringValue):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value: str = ""

    def getValue(self) -> str:
        """ """
        return self.value

    def validate(self) -> bool:
        """
        Makes sure that the filename is valid as a file.
        """
        return os.path.isfile(self.value)
