import configparser
from typing import List

from Granny.Analyses.Parameter import IntParam, Param, StringParam
from Granny.Models.IO.MetaDataIO import MetaDataIO


class MetaDataFile(MetaDataIO):
    def __init__(self, filepath: str):
        MetaDataIO.__init__(self, filepath)

    def load(self) -> List[Param]:
        """ """
        # Parses config file with configparser
        config = configparser.ConfigParser()
        config.read(self.filepath)

        # Gets a list of analysis parameters for the experiment
        params: List[Param] = []
        analysis_args = config["Analysis"]
        analysis_name = StringParam("name", "analysis_name", "")
        analysis_name.setValue(analysis_args["analysis_name"])
        threshold = IntParam("th", "threshold", "")
        threshold.setMax(int(analysis_args["upper_bound"]))
        threshold.setMin(int(analysis_args["lower_bound"]))
        params.extend([analysis_name, threshold])
        return params

    def save(self, params: List[Param]):
        """ """
        pass
