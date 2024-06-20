from Granny.Models.Values.Value import Value


class FileNameValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str

    def validate(self) -> bool:
        """
        Makes sure that the filename is valid as either a file or a directory.
        """
        return True

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
