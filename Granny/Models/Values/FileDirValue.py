import os

from Granny.Models.Values.Value import Value


class FileDirValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value: str = ""

    def setValue(self, value: str):
        """ """
        self.value = value
        if not self.validate():
            ValueError("Not a directory. Please specify a directory.")
        os.makedirs(self.value, exist_ok=True)

    def validate(self) -> bool:
        """
        Checks that the value provided is a valid directory on the file system

        @returns boolean
            returns True if the directory is valid, False otherwise.
        """
        return os.path.isdir(self.value)
