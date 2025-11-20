from typing import Any, List

from Granny.Models.Values.Value import Value


class StringValue(Value):
    """
    Class for a string value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        Initializes a new StringValue parameter.

        Args:
            name (str): Internal name for the parameter.
            label (str): Human-readable label for display purposes.
            help (str): Help description or usage instructions for the parameter.
        """
        super().__init__(name, label, help)
        self.valid_values: List[str] = []

    def setValue(self, value: str):
        """
        Sets the string value if it passes validation.

        Args:
            value (str): The value to assign to this parameter.

        Notes:
            - If the value is not valid (type mismatch or not in `valid_values`),
              it sets the value to None.
            - Marks the value as set using `self.is_set = True`.
        """
        self.value = value if self.validate(value) else None
        self.is_set = True

    def setValidValues(self, values: List[str]):
        """
        Provides a list of valid values for this string paratmer.
        """
        self.valid_values = values

    def getValidValues(self) -> List[str]:
        """
        Gets the list of valid values for this string value.
        """
        return self.valid_values

    def validate(self, value: Any) -> bool:
        """
        {@inheritdoc}
        """
        if self.valid_values != [] and value not in self.valid_values:
            return False
        if type(value) is not str:
            return False
        return True
