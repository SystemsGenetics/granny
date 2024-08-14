from typing import Any, List

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
        self.valid_values: List[str] = []

    def setValue(self, value: str):
        """
        {@inheritdoc}
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
