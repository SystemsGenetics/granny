import os
from typing import Any

from Granny.Models.Values.Value import Value


class FileDirValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value: str = ""

    def setDefaultValue(self, value: Any):
        """ """
        return super().setDefaultValue(value)

    def getDefaultValue(self) -> Any:
        """s"""
        return super().getDefaultValue()

    def setValue(self, value: str):
        """ """
        self.value = value

    def validate(self) -> bool:
        """
        Checks that the value provided is a valid directory on the file system

        @returns boolean
            returns True if the directory is valid, False otherwise.
        """
        return os.path.isdir(self.value)
