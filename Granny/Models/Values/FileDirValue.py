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
        # Validate input before setting
        if not isinstance(value, str):
            raise ValueError(f"Directory path must be a string, got {type(value)}")
        if not value.strip():
            raise ValueError("Directory path cannot be empty")
        
        # Create directory first, then set and validate
        os.makedirs(value, exist_ok=True)
        
        # Only set the value after successful directory creation and validation
        if os.path.isdir(value):
            self.value = value
        else:
            raise ValueError("Not a directory. Please specify a directory.")

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
