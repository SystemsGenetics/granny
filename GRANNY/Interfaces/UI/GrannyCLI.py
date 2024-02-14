import argparse
import os
from argparse import ArgumentParser
from typing import List

from GRANNY.Analyses.Analysis import Analysis
from GRANNY.Analyses.SuperficialScald import SuperficialScald
from GRANNY.Analyses.BlushColor import BlushColor
from GRANNY.Analyses.PeelColor import PeelColor
from GRANNY.Analyses.Segmentation import Segmentation
from GRANNY.Analyses.StarchArea import StarchArea
from GRANNY.Interfaces.UI.GrannyUI import GrannyUI
from GRANNY.Models.Images.RGBImage import RGBImage


class GrannyCLI(GrannyUI):
    def __init__(self, parser: ArgumentParser):
        """
        {@inheritdoc}
        """
        GrannyUI.__init__(self, parser)
        self.image_dir: str = ""
        self.metadata_dir: str = ""
        self.result_dir: str = "results/"
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
        # makes sure that the anaylsis name matches a class.
        return True

    def run(self):
        """
        {@inheritdoc}
        """
        # Get the input arguments.
        self.addArgs()
        args = self.parser.parse_args()
        self.image_dir = args.dir
        self.result_dir = args.result
        # self.metadata_dir = args.metadata
        self.analysis = args.analysis

        # Checks the incoming arguments for errors, if all is okay
        # then collect the arguments.
        if not self.checkArgs():
            exit(1)

        # Get the list of images.
        image_files: List[str] = os.listdir(self.image_dir)
        images = [RGBImage(os.path.join(self.image_dir, image_file)) for image_file in image_files]

        # Iterates through all of the available analysis classes.
        # Finds then one whose machine name matches the argument
        # provided by the user and run the performAnalysis() function.
        analyses = Analysis.__subclasses__()
        for aclass in analyses:
            if self.analysis == aclass.__analysis_name__:
                # call aclass.getParams() and add an addiitonal set of
                # arguments for this class.
                analysis = aclass(images)
                analysis.performAnalysis()

    def addArgs(self) -> None:
        """
        A helper function to parse the command-line arguments into variables.
        """
        self.parser.add_argument(
            "-a",
            "--analysis",
            dest="analysis",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify an analysis you want to perform.",
        )
        self.parser.add_argument(
            "-d",
            "--image_dir",
            dest="dir",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a folder for input images.",
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
            help="Optional. Specify a folder to save results. Default directory is 'results/'.",
        )
