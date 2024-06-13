from typing import List

from Granny.Analyses.Values import Value
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        MetaDataIO.__init__(self, filepath)

    def load(self) -> List[Value]:
        """ """
        return []

    def save(self, params: List[Value]):
        """ """
        return
