import os

from Granny.Models.Values.Value import Value


class FileNameValue(Value):
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

    def setValue(self, value: str):
        """
        {@inheritdoc}
        """
        self.value = value if self.validate() else None
        self.is_set = True

    def validate(self) -> bool:
        """
        Makes sure that the filename is valid as a file.
        """
        return os.path.isfile(self.value)
