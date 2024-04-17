import math
from abc import ABC, abstractmethod
from typing import Any, List


class Param(ABC):
    """
    The base abstract class for a parameter used by Granny.

    This class is used by all Analysis objects for representing the
    parameters that can be used to set how the analysis will perform.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        Instantiates a Param object.

        @param str name
            The machine readable name for this parameter.
        @param str label
            The human readable name for this parameter.
        @param str help
            The help text that is displayed to the user that
            describes how the paramter is used.
        """
        self.help = help
        self.name = name
        self.label = label
        self.default_value = None
        self.type = Any
        self.value = None
        self.is_set = False

    @abstractmethod
    def validate(self) -> bool:
        """
        Validates the value matches the parameter constraints.
        """
        pass

    def getLabel(self) -> str:
        """
        Gets the human readable label for this parameter
        """
        return self.label

    def getHelp(self) -> str:
        return self.help

    def getType(self):
        """
        Gets the Python type for this parameter.
        """
        return self.type

    def setValue(self, value: Any):
        """
        Sets the current value of the paramter.
        """
        self.value = value
        self.is_set = True

    def isSet(self):
        """
        Indicates if the user set this parameter.

        If the paramter is not set then the getValue() function
        will return the default value.
        """
        return self.is_set

    def getValue(self) -> Any:
        """
        Returns the current value of the paramter.
        """
        if not self.is_set:
            return self.default_value
        else:
            return self.value

    @abstractmethod
    def getDefaultValue(self) -> Any:
        """
        Gets the default value for the parameter.
        """
        pass

    @abstractmethod
    def setDefaultValue(self, value: Any):
        """
        Gets the default value for the parameter.
        """
        pass


class NumericParam(Param):
    """
    The base parameter class for a numeric paramter type
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        Param.__init__(self, name, label, help)
        self.type = Any
        self.max_value = math.inf
        self.min_value = -math.inf

    @abstractmethod
    def setMax(self, value: Any):
        """
        Sets the maximum numeric value that this paramter can be set to.
        """
        pass

    @abstractmethod
    def setMin(self, value: Any):
        """
        Sets the minimum numeric value that this parameter can be set to.
        """
        pass


class BoolParam(Param):
    """
    Class for a boolean paramter.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        Param.__init__(self, name, label, help)
        self.type = bool

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True

    def getDefaultValue(self) -> bool:
        """
        {@inheritdoc}
        """
        return self.default_value

    def setDefaultValue(self, value: bool):
        """
        {@inheritdoc}
        """
        self.default_value = value

    def setValue(self, value: bool):
        """
        {@inheritdoc}
        """
        self.value = value


class IntParam(NumericParam):
    """
    Class for an integer paramter.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        NumericParam.__init__(self, name, label, help)
        self.type = int
        self.value_values: list[int] = []
        self.value: int = 0

    def setMax(self, value: int):
        """
        {@inheritdoc}
        """
        self.max_value = value

    def setMin(self, value: int):
        """
        {@inheritdoc}
        """
        self.min_value = value

    def setDefaultValue(self, value: int):
        """
        Gets the default value for the parameter.
        """
        self.default_value = value

    def setValidValues(self, values: list[int]):
        """
        Provides a list of valid values for this integer parameter.
        """
        self.valid_values = values

    def setValue(self, value: int):
        """
        {@inheritdoc}
        """
        self.value = value

    def getValidValues(self) -> List[int]:
        """
        Gets the list of valid values for this integer parameter.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True

    def getDefaultValue(self) -> int:
        """
        {@inheritdoc}
        """
        return self.default_value


class FloatParam(NumericParam):
    """
    Class for a float parameter.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        NumericParam.__init__(self, name, label, help)
        self.type = float
        self.value_values: List[float] = []
        self.value: float = 0

    def setMax(self, value: float):
        """
        {@inheritdoc}
        """
        self.max_value = value

    def setMin(self, value: float):
        """
        {@inheritdoc}
        """
        self.min_value = value

    def setDefaultValue(self, value: float):
        """
        Sets the default value for the parameter.
        """
        self.default_value = value

    def getDefaultValue(self) -> float:
        """
        Gets the default value for the parameter.
        """
        return self.default_value

    def setValidValues(self, values: List[float]):
        """
        Provides a list of valid values for this integer parameter.
        """
        self.valid_values = values

    def setValue(self, value: float):
        """
        {@inheritdoc}
        """
        super().setValue(value)

    def getValidValues(self) -> List[float]:
        """
        Gets the list of valid values for this integer parameter.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True


class StringParam(Param):
    """
    Class for a string parameter.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        Param.__init__(self, name, label, help)
        self.type = str
        self.value_values: List[str] = []

    def setValidValues(self, values: List[str]):
        """
        Provides a list of valid values for this string paratmer.
        """
        self.valid_values = values

    def getValidValues(self) -> List[str]:
        """
        Gets the list of valid values for this string paramter.
        """
        return self.value_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
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
