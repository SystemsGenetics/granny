"""
Base abstract Analysis class for the analyses to be called by either the command line interface
or the graphical user interface.

Author: Nhan Nguyen
Date: July 12, 2024
"""

import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List
from multiprocessing import Pool

from Granny.Models.Images.Image import Image
from Granny.Models.Values.IntValue import IntValue
from Granny.Models.Values.StringValue import StringValue
from Granny.Models.Values.Value import Value


class Analysis(ABC):

    __analysis_name__ = "analysis"

    def __init__(self):
        """
        Abstract base class for different types of analyses. This class provides the structure
        and common functionality for performing analyses, including handling input parameters,
        return values, and metadata.

        Attributes:
            in_params (Dict[str, Value]): Dictionary to store input parameters for the analysis.
            ret_values (Dict[str, Value]): Dictionary to store return values from the analysis.
            compatibility (Dict[str, Dict[str, str]]): Dictionary to define compatibility with other analyses.
            metadata (List[Value]): List to store metadata about the analysis.
        """
        # The list of INPUT parameter values for the analysis.
        self.in_params: Dict[str, Value] = {}

        # The list of RETURN values for the analysis.
        self.ret_values: Dict[str, Value] = {}

        # # The set of over analyses that are compatible with this analysis.
        # # it should be a list of key/value pairs, where the top-level key
        # # is the other analysis to which this one is compatible. It's value
        # # is a list that maps input parameters of this analysis to return
        # # values from the compatible analysis.
        self.compatibility: Dict[str, Dict[str, str]] = {}

        # Stores metadata about the analysis. These values
        # will get added to the result images.
        self.metadata: List[Value] = []

        # Set some default metadata values for all analyses:
        # The analysis date and time.
        time = StringValue(
            "dt", "datetime", "Date and time of when the analysis was performed."
        )
        time.setValue(datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.metadata.append(time)

        # The analysis id.
        id = StringValue("id", "identifier", "Unique identifier for the analysis")
        id.setValue(str(uuid.uuid4()))
        self.metadata.append(id)

        # Current directory
        path = StringValue(
            "path", "cur_dir", "Absolute file path of the current directory."
        )
        path.setValue(os.path.abspath(os.curdir))
        self.metadata.append(path)

        # Number of CPU cores for parallel processing
        self.cpu = IntValue(
            "cpu",
            "cpu",
            "Number of CPU cores to use for parallel processing. "
            "Set to 0 for automatic (uses 80% of available cores). "
            "Default is 0 (automatic)."
        )
        self.cpu.setMin(0)
        self.cpu.setMax(os.cpu_count() or 1)
        self.cpu.setValue(0)  # 0 = auto mode
        self.cpu.setIsRequired(False)
        self.addInParam(self.cpu)

    def addInParam(self, *params: Value):
        """
        Adds one or more parameters to the input parameter dictionary.

        Args:
            *params (Value): One or more Value instances to add as input parameters.
        """
        for param in params:
            self.in_params[param.getName()] = param

    def getInParams(self) -> Dict[str, Value]:
        """
        Retrieves all input parameters for the analysis.

        Returns:
            Dict[str, Value]: A dictionary of input parameters.
        """
        return dict(self.in_params)

    def resetInParams(self):
        """
        Resets the list of input parameters, clearing all current input parameters.
        """
        self.in_params = {}

    def addRetValue(self, *values: Value):
        """
        Adds one or more values to the return value dictionary.

        Args:
            *values (Value): One or more Value instances to add as return values.
        """
        for value in values:
            self.ret_values[value.getName()] = value

    def getRetValues(self) -> Dict[str, Value]:
        """
        Retrieves all return values from the analysis.

        Returns:
            Dict[str, Value]: A dictionary of return values.
        """
        return dict(self.ret_values)

    def resetRetValues(self):
        """
        Resets the list of return values, clearing all current return values.
        """
        self.ret_values = {}

    def _parse_qr_from_filename(self, filename: str) -> dict:
        """
        Extract QR code information from segmented image filename.

        Expected format: PROJECT_LOT_DATE_VARIETY_fruit_##.png
        Example: APPLE2025_LOT001_2025-12-02_BB-Late_fruit_01.png

        Args:
            filename: Image filename (with or without path)

        Returns:
            Dictionary with QR information:
            {
                'project': project code or empty string,
                'lot': lot code or empty string,
                'date': date string or empty string,
                'variety': variety string or empty string
            }

        Notes:
            - Returns empty strings for all fields if parsing fails
            - Handles legacy filenames gracefully (no QR data)
        """
        import re
        from pathlib import Path

        # Extract just the filename without path
        filename_only = Path(filename).name

        # Pattern: PROJECT_LOT_DATE_VARIETY_fruit_##.png
        # Use regex to match everything before "_fruit_##"
        pattern = r'^(.+?)_(.+?)_(.+?)_(.+?)_fruit_\d+\.(?:png|jpg|jpeg)$'
        match = re.match(pattern, filename_only)

        if match:
            return {
                'project': match.group(1),
                'lot': match.group(2),
                'date': match.group(3),
                'variety': match.group(4)
            }
        else:
            # Parsing failed - return empty strings (no QR data)
            return {
                'project': '',
                'lot': '',
                'date': '',
                'variety': ''
            }

    def _add_qr_metadata(self, result_img, filename: str):
        """
        Parse QR/barcode metadata from filename and add to result image.

        Args:
            result_img: Image instance to add metadata values to
            filename: Image filename to parse
        """
        qr_info = self._parse_qr_from_filename(filename)
        if qr_info['project']:
            project_val = StringValue("project", "project", "Project code from QR code")
            project_val.setValue(qr_info['project'])
            result_img.addValue(project_val)

            lot_val = StringValue("lot", "lot", "Lot code from QR code")
            lot_val.setValue(qr_info['lot'])
            result_img.addValue(lot_val)

            date_val = StringValue("date", "date", "Date from QR code")
            date_val.setValue(qr_info['date'])
            result_img.addValue(date_val)

            variety_val = StringValue("variety", "variety", "Variety from QR code")
            variety_val.setValue(qr_info['variety'])
            result_img.addValue(variety_val)

    def performAnalysis(self) -> List[Image]:
        """
        Once all required parameters have been set, this function is used
        to perform the analysis.
        """
        # initiates user's input
        self.input_images: ImageListValue = self.in_params.get(self.input_images.getName())  # type: ignore

        # initiates Granny.Model.Images.Image instances for the analysis using the user's input
        self.input_images.readValue()
        self.images = self.input_images.getImageList()

        # Allow the child module to set up it's member variables, etc.
        self._preRun()

        # perform analysis with multiprocessing
        num_cpu = os.cpu_count() or 1
        user_cpu = self.cpu.getValue()

        if user_cpu == 0:
            # Auto mode: use 80% of available cores
            cpu_count = int(num_cpu * 0.8) or 1
        else:
            # User-specified: don't exceed available cores
            cpu_count = min(user_cpu, num_cpu)

        with Pool(cpu_count) as pool:
            results = pool.map(self._processImage, self.images)

        # Allow the child module to perform post processing after
        # all images have been processed.
        return self._postRun(results)


    @abstractmethod
    def _preRun(self):
        pass

    @abstractmethod
    def _postRun(self, results):
        pass

    @abstractmethod
    def _processImage(self, image_instance: Image) -> Image:
        pass
