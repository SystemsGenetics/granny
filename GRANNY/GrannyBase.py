import argparse

from Interfaces.UI.GrannyUI import GrannyUI


class GrannyBase(object):
    def __init__(self):
        self.granny_version: str = None
        self.interface: GrannyUI = None

    def run(self):
        pass


def cli():
    parser = argparse.ArgumentParser(description="")

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
        "-r",
        "--result_dir",
        dest="result",
        type=str,
        nargs="?",
        required=False,
        help="Optional. Specify a folder to save results. Default directory is 'results/'.",
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


