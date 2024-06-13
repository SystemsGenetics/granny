"""
Base abstract Analysis class for the analyses to be called by either the command line interface
or the graphical user interface.

Author: Nhan Nguyen
Date: May 21, 2024
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from Granny.Analyses.Values import Value, StringValue
from Granny.Models.Images.Image import Image
from Granny.Models.Images.MetaData import MetaData
from Granny.Models.Images.RGBImage import RGBImage


class Analysis(ABC):

    __analysis_name__ = "analysis"

    def __init__(self):
        """
        Initializes an instance of an Analysis object

        @param GRANNY.Models.Images.Image An instance of an Image object

        @return GRANNY.Analyses.Analysis.Analysis object.
        """
        # The list of input parameter values for the analysis.
        self.params: Dict[str, Value] = {}

        # The list of return values fro the analysis.
        self.ret_values: Dict[str, Value] = {}

        # The set of over analyses that are compatible with this analysis.
        # it should be a list of key/value pairs, where the top-level key
        # is the other analysis to which this one is compatible. It's value
        # is a list that maps input parameters of this analysis to return
        # values from the compatible analysis.
        self.compatibility: Dict[str, Dict] = {}

        # Stores metadata about the analysis. These values
        # will get added to resulting images.
        self.metadata: List[Value] = []

        # Set some default metadata values for all analyses:
        # The analysis date and time.
        time = StringValue("dt", "datetime", "Date and time of when the analysis was performed.")
        time.setDefaultValue(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.metadata.append(time)

        # The analysis id.
        id = StringValue("id", "identifier", "Unique identifier for the analysis")
        id.setDefaultValue(str(uuid.uuid4()))
        self.metadata.append(id)

    def addParam(self, *params: Value):
        """
        Adds a parameter to the parameter dictionary
        """
        for param in params:
            self.params[param.getName()] = param

    def getParams(self) -> Dict[str, Value]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return dict(self.params)

    def addRetValue(self, *values: Value):
        """
        Adds a parameter to the parameter dictionary
        """
        for value in values:
            self.ret_values[value.getName()] = value

    def getRetValues(self) -> Dict[str, Value]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return dict(self.ret_values)


    @abstractmethod
    def performAnalysis(self) -> Any:
        """
        Once all required parameters have been set, this function is used
        to perform the analysis.
        """
        pass
