from typing import List, Any

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

    def validate(self, value: Any) -> bool:
        """
        {@inheritdoc}
        """
        if not isinstance(value, float):
            return False
        if self.valid_values and value not in self.valid_values:
            return False
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True
