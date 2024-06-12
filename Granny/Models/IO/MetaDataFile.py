from typing import List

from Granny.Analyses.Parameter import Param
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        MetaDataIO.__init__(self, filepath)

    def load(self) -> List[Param]:
        """ """
        return []

    def save(self, params: List[Param]):
        """ """
        return
