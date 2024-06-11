from typing import List

from Granny.Analyses.Parameter import Param
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaData(object):
    def __init__(self):
        self.params: List[Param] = []

    def updateParameters(self, params: List[Param]):
        """
        Adds a list of parameters to self.params
        """
        self.params.extend(params)

    def getParameters(self):
        """
        Gets the list of parameters of the experiment
        """
        return self.params

    def save(self, metadata_io: MetaDataIO):
        """
        Calls MetaDataIO to write parameters
        """
        metadata_io.save(self.params)
