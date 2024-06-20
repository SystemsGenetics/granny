from typing import List

from Granny.Models.Values.NumericValue import NumericValue


class FloatValue(NumericValue):
    """
    Class for a float value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = float
        self.valid_values: List[float] = []
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
        Sets the default value for the value.
        """
        self.default_value = value

    def getDefaultValue(self) -> float:
        """
        Gets the default value for the value.
        """
        return self.default_value

    def setValidValues(self, values: List[float]):
        """
        Provides a list of valid values for this integer value.
        """
        self.valid_values = values

    def getValidValues(self) -> List[float]:
        """
        Gets the list of valid values for this integer value.
        """
        return self.valid_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True
