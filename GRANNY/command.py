import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse

from GRANNY import (
    GRANNY_BlushColor,
    GRANNY_PeelColor,
    GRANNY_Segmentation,
    GRANNY_StarchIndex,
    GRANNY_SuperficialScald,
)


def main():
    parser = argparse.ArgumentParser(
        description="Implementation of Mask-RCNN and image binarization to rate disorder severity on Granny Smith apples."
    )

    parser.add_argument(
        "-a",
        "--action",
        dest="action",
        type=str,
        nargs="?",
        required=True,
        help="Required. Specify an action to perform.",
    )

    parser.add_argument(
        "-d",
        "--image_dir",
        dest="dir",
        type=str,
        nargs="?",
        required=True,
        help="Required. Specify a directory or a file.",
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

    args = parser.parse_args()

    if args.action == "extract":
        GRANNY_Segmentation.GrannySegmentation(
            args.action, args.dir, args.num_instances
        ).extract_instances_with_MaskRCNN()
    elif args.action == "scald":
        GRANNY_SuperficialScald.GrannySuperficialScald(
            args.action, args.dir, args.num_instances
        ).rate_GrannySmith_superficial_scald()
    elif args.action == "peel":
        GRANNY_PeelColor.GrannyPeelColor(
            args.action, args.dir, args.num_instances
        ).extract_green_yellow_values()
    elif args.action == "starch":
        GRANNY_StarchIndex.GrannyStarchIndex(args.action, args.dir, args.num_instances).main()
    elif args.action == "blush":
        GRANNY_BlushColor.GrannyPearBlush(args.action, args.dir, args.num_instances).main()
    else:
        print("\t- Invalid Action. -")
