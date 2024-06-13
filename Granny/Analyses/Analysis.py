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
from typing import Dict, List, Any

from Granny.Analyses.Parameter import Param, StringParam
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
        self.in_params: Dict[str, Param] = {}
        self.out_params: Dict[str, Param] = {}

        self.compatibility = {}

    def setInParam(self, params: Dict[str, Param]) -> None:
        """
        Sets the parameter dictionary
        """
        self.in_params = params

    def addInParam(self, *params: Param):
        """
        Adds a parameter to the parameter dictionary
        """
        for param in params:
            self.in_params[param.getName()] = param

    def getInParams(self) -> Dict[str, Param]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return dict(self.in_params)

    def setOutParam(self, params: Dict[str, Param]) -> None:
        """
        Sets the parameter dictionary
        """
        self.out_params = params

    def addOutParam(self, *params: Param):
        """
        Adds a parameter to the parameter dictionary
        """
        for param in params:
            self.out_params[param.getName()] = param

    def getOutParams(self) -> Dict[str, Param]:
        """
        Returns to the GUI/CLI all the required parameters in self.params
        """
        return dict(self.out_params)
        
    def getDefaultMetadata(self):
        """
        Generates general metadata for the analysis, including: date and time, analysis name, id.
        """
        # the analysis date and time
        time = StringParam("dt", "datetime", "Date and time of when the analysis was performed.")
        time.setDefaultValue(datetime.now().strftime("%Y-%m-%d %H:%M"))

        # the analysis id
        id = StringParam("id", "identifier", "Unique identifier for the analysis")
        id.setDefaultValue(str(uuid.uuid4()))

        self.addParam(time, id)

    @abstractmethod
    def performAnalysis(self) -> Any:
        """
        Once all required parameters have been set, this function is used
        to perform the analysis.
        """
        pass
