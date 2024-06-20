from typing import List

from Granny.Models.Values.Value import Value


class StringValue(Value):
    """
    Class for a string value.
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = str
        self.value_values: List[str] = []

    def setValidValues(self, values: List[str]):
        """
        Provides a list of valid values for this string paratmer.
        """
        self.valid_values = values

    def getValidValues(self) -> List[str]:
        """
        Gets the list of valid values for this string value.
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
