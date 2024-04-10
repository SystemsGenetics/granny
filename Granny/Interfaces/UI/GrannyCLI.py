import argparse
import os
from argparse import ArgumentParser
from typing import List

from Granny.Analyses.Analysis import Analysis
from Granny.Analyses.BlushColor import BlushColor
from Granny.Analyses.PeelColor import PeelColor
from Granny.Analyses.Segmentation import Segmentation
from Granny.Analyses.StarchArea import StarchArea
from Granny.Analyses.SuperficialScald import SuperficialScald
from Granny.Interfaces.UI.GrannyUI import GrannyUI
from Granny.Models.Images.Image import Image
from Granny.Models.Images.RGBImage import RGBImage


class GrannyCLI(GrannyUI):
    def __init__(self, parser: ArgumentParser):
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)
        self.image_dir: str = ""
        self.metadata_dir: str = ""
        self.result_dir: str = ""
        self.analysis: str = ""

    def checkArgs(self) -> bool:
        """
        Checks the incoming command-line arguments.

        Ensures that the command-line arguments are appropriate for the
        image collection and analyses to be performed.  When arguments
        do not pass a test, then error messages are printed to STDERR
        and False is returned.

        @return book
          True if all argument are good, FALSE otherwsie.
        """
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                return True
        return False

    def listImages(self, image_dir: str) -> List[Image]:
        """
        Reads the input image directory and returns a list of Granny.Model.Image instances
        """
        # Gets the list of files in the directory.
        image_files: List[str] = os.listdir(image_dir)
        IMAGE_EXTENSION = (
            ".JPG",
            ".JPG".lower(),
            ".PNG",
            ".PNG".lower(),
            ".JPEG",
            ".JPEG".lower(),
            ".TIFF",
            ".TIFF".lower(),
        )

        # Creates a list of Image instances
        images: List[Image] = []
        for image_file in image_files:
            if image_file.endswith(IMAGE_EXTENSION):
                images.append(RGBImage(os.path.join(self.image_dir, image_file)))
        return images


    def run(self):
        """
        {@inheritdoc}
        """
        # Get the input arguments.
        self.addAnalysisArgs()
        analysis_args, _ = self.parser.parse_known_args()
        self.image_dir = analysis_args.dir
        self.result_dir = analysis_args.result
        # self.metadata_dir = args.metadata
        self.analysis = analysis_args.analysis

        # Checks the incoming arguments for errors, if all is okay then collect the arguments.
        if not self.checkArgs():
            exit(1)

        # Gets Image instances
        images = self.listImages(self.image_dir)

        # Gets parameter arguments
        self.addParameterArgs()
        param_args, _ = self.parser.parse_known_args()

        # Iterates through all of the available analysis classes.
        # Finds then one whose machine name matches the argument
        # provided by the user and run the performAnalysis() function.
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                # call aclass.getParams() and add an additional set of
                # arguments for this class.
                analysis = aclass(images)
                analysis.performAnalysis()

    def addAnalysisArgs(self) -> None:
        """
        Parses the command-line analysis arguments: analysis, image directory, metadata directory,
        and result directory
        """
        self.parser.add_argument(
            "-a",
            "--analysis",
            dest="analysis",
            type=str,
            nargs="?",
            required=True,
            choices=["segmentation", "blush", "color", "scald", "starch"],
            help="Required. Specify an analysis you want to perform.",
        )
        self.parser.add_argument(
            "-d",
            "--image_dir",
            dest="dir",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a folder containing input images.",
        )
        # self.parser.add_argument(
        #     "-m",
        #     "--metadata",
        #     dest="metadata",
        #     type=str,
        #     nargs="?",
        #     required=True,
        #     help="Required. Specify a path for metadata file.",
        # )
        self.parser.add_argument(
            "-r",
            "--result_dir",
            dest="result",
            type=str,
            nargs="?",
            required=False,
            default="results/",
            help="Optional. Specify a folder to save results. Default directory is 'results/'.",
        )

    def addParameterArgs(self) -> None:
        """
        Parses the command-line parameter arguments: type, default, upper bound, lower bound,
        valid values, and label
        """
        self.parser.add_argument(
            "--type",
            dest="type",
            type=str,
            nargs="?",
            required=False,
            help="",
        )
        self.parser.add_argument(
            "--default",
            dest="default",
            type=str,
            nargs="?",
            required=False,
            help="",
        )
        self.parser.add_argument(
            "--upper",
            dest="upper",
            type=int,
            nargs="?",
            required=False,
            help="",
        )
        self.parser.add_argument(
            "--lower",
            dest="lower",
            type=int,
            nargs="?",
            required=False,
            help="",
        )
        self.parser.add_argument(
            "--valid",
            dest="valid",
            type=int,
            nargs="+",
            required=False,
            help="Provides multiple valid values for the analysis",
        )
        self.parser.add_argument(
            "--label",
            dest="label",
            type=str,
            nargs="?",
            required=False,
            help="",
        )
