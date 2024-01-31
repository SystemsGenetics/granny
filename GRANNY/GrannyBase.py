import argparse

from GRANNY.Analyses.BlushColor import BlushColor
from GRANNY.Analyses.PeelColor import PeelColor
from GRANNY.Analyses.Segmentation import Segmentation
from GRANNY.Analyses.StarchArea import StarchArea
from GRANNY.Analyses.SuperficialScald import SuperficialScald
from GRANNY.Interfaces.UI.GrannyCLI import GrannyCLI
from GRANNY.Interfaces.UI.GrannyUI import GrannyUI
from GRANNY.Models.Images.Image import Image


class GrannyBase(object):
    def __init__(self):
        self.granny_version: str = "1.0"
        self.interface: GrannyUI = GrannyCLI()

    def run(self):
        args = self.cli()
        if args.action == "segmentation":
            print("this is segmentation analysis")
            # Segmentation()
        elif args.action == "scald":
            print("this is superficial scald analysis")
            # SuperficialScald()
        elif args.action == "peel":
            print("this is peel color analysis")
            # PeelColor()
        elif args.action == "starch":
            print("this is starch percentage analysis")
            # StarchArea()
        elif args.action == "blush":
            print("this is blush color analysis")
            # BlushColor()
        else:
            print("\t- Invalid Action. -")

    def cli(self):
        parser = argparse.ArgumentParser(description="Granny v1.0")

        parser.add_argument(
            "-d",
            "--image_dir",
            dest="dir",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a folder for input images.",
        )
        parser.add_argument(
            "-m",
            "--metadata",
            dest="metadata",
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a path for metadata file.",
        )
        parser.add_argument(
            "-a",
            "--analysis",
            dest="analysis",
            type=int,
            nargs="?",
            required=True,
            help="Required. Specify an analysis you want to perform.",
        )
        parser.add_argument(
            "-r",
            "--result_dir",
            dest="result",
            type=str,
            nargs="?",
            required=False,
            help="Optional. Specify a folder to save results. Default directory is 'results/'.",
        )

        return parser.parse_args()


def execute():
    GrannyBase().run()
