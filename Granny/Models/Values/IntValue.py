from typing import List, Any

from Granny.Models.Values.NumericValue import NumericValue


class IntValue(NumericValue):
    """
    Class for an integer value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = int
        self.valid_values: List[int] = []
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

    def setValidValues(self, values: list[int]):
        """
        Provides a list of valid values for this integer value.
        """
        self.valid_values = values

    def setValue(self, value: Any):
        """
        Sets the current value of the value.
        """
        self.validate(value)
        self.value = value
        self.is_set = True
        self.setType(type(value))

    def getValidValues(self) -> List[int]:
        """
        Gets the list of valid values for this integer value.
        """
        return self.valid_values

    def validate(self, value: Any) -> bool:
        """
        {@inheritdoc}
        """
        if self.valid_values != [] and value not in self.valid_values:
            return False
        if self.min_value > value or self.max_value < value:
            return False
        if type(value) is not int:
            return False
        return True
