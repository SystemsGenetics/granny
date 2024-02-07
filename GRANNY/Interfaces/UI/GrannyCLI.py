import argparse
import os

from GRANNY.Analyses.StarchArea import StarchArea
from GRANNY.Analyses.SuperficialScald import SuperficialScald
from GRANNY.Interfaces.UI.GrannyUI import GrannyUI
from GRANNY.Models.Images.RGBImage import RGBImage


class GrannyCLI(GrannyUI):
    def __init__(self):
        GrannyUI.__init__(self)
        self.image_dir: str = ""
        self.metadata_dir: str = ""
        self.result_dir: str = ""
        self.analysis: str = ""

    def printHelp(self):
        os.system("granny-cli --help")

    def checkArgs(self):
        # from params in analyses
        pass

    def run(self):
        self.cli()
        image = RGBImage(self.image_dir)
        if self.analysis == "segmentation":
            print("this is segmentation analysis")
            # Segmentation()
        elif self.analysis == "scald":
            SuperficialScald(image).performAnalysis()
        elif self.analysis == "peel":
            print("this is peel color analysis")
            # PeelColor()
        elif self.analysis == "starch":
            StarchArea(image).performAnalysis()
        elif self.analysis == "blush":
            print("this is blush color analysis")
            # BlushColor()
        else:
            print("\t- Invalid Action. -")

    def cli(self):
        parser = argparse.ArgumentParser(description="Welcome to Granny v1.0")
        parser.add_argument(
            "-a",
            "--analysis",
            dest="analysis",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify an analysis you want to perform.",
        )
        parser.add_argument(
            "-d",
            "--image_dir",
            dest="dir",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a folder for input images.",
        )
        # parser.add_argument(
        #     "-m",
        #     "--metadata",
        #     dest="metadata",
        #     type=str,
        #     nargs="?",
        #     required=True,
        #     help="Required. Specify a path for metadata file.",
        # )
        parser.add_argument(
            "-r",
            "--result_dir",
            dest="result",
            type=str,
            nargs="?",
            required=False,
            help="Optional. Specify a folder to save results. Default directory is 'results/'.",
        )

        args = parser.parse_args()
        self.image_dir = args.dir
        self.result_dir = args.result
        # self.metadata_dir = args.metadata
        self.analysis = args.analysis
