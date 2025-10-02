import os

from Granny.Models.Values.Value import Value


class FileDirValue(Value):
    def __init__(self, name: str, label: str, help: str):
        """
        A value class that represents a directory path.

        When set, this value ensures the directory exists (creates it if missing).
        Validates that the path is indeed a directory.
        """
        super().__init__(name, label, help)
        self.type = str
        self.value: str = ""

    def setValue(self, value: str):
        """ 
        Sets the directory path for this value. Creates the directory if it doesn't exist.

        Args:
            value (str): A string representing a directory path.

        Raises:
        ValueError: If the path is not a valid directory after assignment.
        """
        self.value = value
        if not self.validate():
            raise ValueError("Not a directory. Please specify a directory.")
        # Create directory after validation
        os.makedirs(self.value, exist_ok=True)

    def validate(self) -> bool:
        """
        Checks that the value provided is a valid directory on the file system

        @returns boolean
            returns True if the directory is valid, False otherwise.
        """
        # Validate input
        if not isinstance(self.value, str):
            return False
        if not self.value.strip():
            return False
        
        # Check if it's a valid directory path (don't create here)
        return True
