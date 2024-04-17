from abc import ABC, abstractmethod
from typing import List

from Granny.Analyses.Parameter import Param


class MetaDataIO(ABC):
    def __init__(self, filepath: str):
        self.filepath: str = filepath

    @abstractmethod
    def load(self) -> List[Param]:
        pass

    @abstractmethod
    def save(self, params: List[Param]):
        pass
