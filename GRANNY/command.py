import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import argparse
from . import GRANNY as granny

def main():

    parser = argparse.ArgumentParser(description = "Implementation of Mask-RCNN and image binarization to rate disorder severity on Granny Smith apples.")

    parser.add_argument("-a", "--action", dest = "action", type = str, nargs = "?", 
    required = True, help = "Required. Specify an action to perform.")

    parser.add_argument("-d", "--image_dir", dest = "dir", type = str, nargs = "?", 
    required = True, help = "Required. Specify a directory or a file.")

    parser.add_argument("-n", "--num_instances", dest = "num_instances", type = int, nargs = "?", 
    required = False, help = "Optional, default is 18. The number of instances on each image.")
    
    parser.add_argument("-rgb", "--rgb_code", dest = "rgb", type = str, nargs = "?", 
    default = "", required = False, help = "Optional. Provide an RGB value for color reference.")

    parser.add_argument("-v", "--verbose", dest = "verbose", type = int, nargs = "?", 
    default = 0, required = False, help = "Optional. Specify 1 to turn on model display.")

    args = parser.parse_args()

    if args.action == "extract": 
        granny.GrannyExtractInstances(args.action, args.dir, args.num_instances, args.verbose).mask_extract_image()
    elif args.action == "scald": 
        granny.GrannySuperficialScald(args.action, args.dir, args.num_instances, args.verbose).rate_binarize_image()
    elif args.action == "peel": 
        granny.GrannyPeelColor(args.action, args.dir, args.num_instances, args.rgb, args.verbose).sort_peel_color()
    elif args.action == "starch": 
        granny.GrannyStarchIndex(args.action, args.dir, args.num_instances, args.verbose).main()
    else: 
        print("\t- Invalid Action. -")

