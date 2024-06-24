from typing import List

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

    def getValidValues(self) -> List[int]:
        """
        Gets the list of valid values for this integer value.
        """
        return self.valid_values

    def validate(self) -> bool:
        """
        {@inheritdoc}
        """
        return True
