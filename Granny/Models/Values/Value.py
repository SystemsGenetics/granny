from abc import ABC, abstractmethod
from typing import Any


class Value(ABC):
    """
    The base abstract class for a values used by Granny.

    This class is used by all Analysis objects for representing the
    values that can be used to set how the analysis will perform.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        Instantiates a Value object.
        """
        self.name = name
        self.label = label
        self.help = help
        self.type = Any
        self.value = None
        self.is_set = False

    @abstractmethod
    def validate(self) -> bool:
        """
        Validates the value matches the value constraints.
        """
        pass

    def getName(self) -> str:
        """
        Gets the machine readable name for this value
        """
        return self.name

    def getLabel(self) -> str:
        """
        Gets the human readable label for this value
        """
        return self.label

    def getHelp(self) -> str:
        return self.help

    def getType(self):
        """
        Gets the Python type for this value.
        """
        return self.type

    def setValue(self, value: Any):
        """
        Sets the current value of the value.
        """
        self.value = value
        self.is_set = True

    def isSet(self):
        """
        Indicates if the user set this value.

        If the value is not set then the getValue() function
        will return the default value.
        """
        return self.is_set

    def getValue(self) -> Any:
        """
        Returns the current value of the value.
        """
        return self.value

    def readValue(self):
        """
        Reads the value from the storage system.
        """
        pass

    def writeValue(self):
        """
        Writes the value to the storage system.
        """
        pass
