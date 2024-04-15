from abc import ABC, abstractmethod


class MetaDataIO(ABC):
    def __init__(self, filepath: str):
        self.filepath: str = filepath

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass
