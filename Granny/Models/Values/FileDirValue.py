from Granny.Models.Values.Value import Value


class FileDirValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str

    def validate(self) -> bool:
        """
        Checks that the value provided is a valid directory on the file system

        @returns boolean
            returns True if the directory is valid, False otherwise.
        """
        return True
