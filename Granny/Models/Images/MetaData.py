from typing import List

from Granny.Analyses.Parameter import Param
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaData(object):
    def __init__(self, metadata_io: MetaDataIO):
        self.io: MetaDataIO = metadata_io
        self.params: List[Param] = self.io.load()

    def setMetaData(self, params: List[Param]):
        """
        Adds a list of parameters to self.params
        """
        self.params.extend(params)

    def getMetaData(self):
        """
        Gets the list of parameters of the experiment
        """
        return self.params

    def saveMetaData(self):
        """
        Calls MetaDataIO to write parameters
        """
        self.io.save(self.params)
