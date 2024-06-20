from abc import ABC, abstractmethod
from typing import List

from Granny.Models.Values.Value import Value


class MetaDataIO(ABC):
    def __init__(self, filepath: str):
        self.filepath: str = filepath

    @abstractmethod
    def load(self) -> List[Value]:
        pass

    @abstractmethod
    def save(self, params: List[Value]):
        pass
