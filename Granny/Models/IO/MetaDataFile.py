from typing import List

from Granny.Models.IO.MetaDataIO import MetaDataIO
from Granny.Models.Values.Value import Value


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        MetaDataIO.__init__(self, filepath)

    def load(self) -> List[Value]:
        """ """
        return []

    def save(self, params: List[Value]):
        """ """
        return
