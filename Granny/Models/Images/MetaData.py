from typing import List

from Granny.Analyses.Parameter import Param
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaData(object):
    def __init__(self, metadata_io: MetaDataIO):
        self.params: List[Param]

    def setMetaData(self, params: List[Param]):
        pass

    def saveMetaData(self):
        """
        Calls MetaDataIO to write parameters
        """
        pass
