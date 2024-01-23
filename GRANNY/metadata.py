import numpy as np

from abc import ABC
from numpy.typing import NDArray


class MetaData(ABC):
    def __init__(self):
        None


class MetaDataIO(object):
    def __init__(self):
        None


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        super(MetaDataIO, self).__init__()
        self.filepath: str = filepath
        self.metadata: NDArray = None

    def load():
        pass

    def save(metadata: MetaData):
        pass
