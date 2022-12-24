import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import argparse
from . import GRANNY as granny

def main():
    gn = granny.GRANNY()

    parser = argparse.ArgumentParser(description = "Implementation of Mask-RCNN and image binarization to rate disorder severity on Granny Smith apples.")

    parser.add_argument("-a", "--action", dest = "action", type = str, nargs = "?", 
    required = True, help = "Required. Specify either \"extract\" or \"rate\".")

    parser.add_argument("-p", "--image_path", dest = "path", type = str, nargs = "?", 
    required = True, help = "Required. Specify a directory for --action==\"extract\" or a file for --action==\"rate\"")

    parser.add_argument("-m", "--mode", dest = "mode", type = int, nargs = "?", 
    required = False, help = "Optional. Specify 2 for multiples images processing when --action==\"rate\".")
    
    parser.add_argument("-v", "--verbose", dest = "verbose", type = int, nargs = "?", 
    default = gn.VERBOSE, required = False, help = "Optional. Specify 0 to switch off model display.")
    
    args = parser.parse_args()
    
    gn.setParameters(args.action, args.path, args.mode)

    gn.setVerbosity(args.verbose)

    gn.main()

def launch_gui(): 
    
    gn = granny.GRANNY()

    gn.launch_gui()

