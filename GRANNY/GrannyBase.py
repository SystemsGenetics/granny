import argparse

from Interfaces.UI.GrannyUI import GrannyUI


class GrannyBase(object):
    def __init__(self):
        self.granny_version: str = None
        self.interface: GrannyUI = None

    def run(self):
        parser = argparse.ArgumentParser(description="")

        parser.add_argument("-h", "--help")
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
            type=str,
            nargs="?",
            required=True,
            help="Required. Specify a path for metadata file.",
        )
        parser.add_argument(
            "-r",
            "--result_dir",
            type=str,
            nargs="?",
            required=True,
            help="Optional. Specify a folder to save results.",
        )
        parser.add_argument(
            "-n",
            "--num_instances",
            dest="num_instances",
            type=int,
            nargs="?",
            required=False,
            help="Optional, default is 18. The number of instances on each image.",
        )
