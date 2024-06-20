import math
from abc import abstractmethod
from typing import Any

from Granny.Models.Values.Value import Value


class NumericValue(Value):
    """
    The base value class for a numeric value type
    """

    def __init__(self, name: str, label: str, help: str):
        """
        {@inheritdoc}
        """
        super().__init__(name, label, help)
        self.type = Any
        self.max_value = math.inf
        self.min_value = -math.inf

    @abstractmethod
    def setMax(self, value: Any):
        """
        Sets the maximum numeric value that this value can be set to.
        """
        pass

    @abstractmethod
    def setMin(self, value: Any):
        """
        Sets the minimum numeric value that this value can be set to.
        """
        pass
