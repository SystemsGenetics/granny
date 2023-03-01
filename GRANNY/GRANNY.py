import cv2
import matplotlib.pyplot as plt
import re
from . import GRANNY_config as config
import skimage
import shutil
import pathlib
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GrannyBaseClass(object):
    """ 
        Implementation for semantic segmentation of instances and superficial scald rating in "Granny Smith" apples
    """

    def __init__(self, action = "", fname = "", num_instances = 1, verbose = 0 ):
        # current directory
        self.ROOT_DIR = pathlib.Path(__file__).parent.resolve()

        # logs
        self.MODEL_DIR = os.path.join(os.path.curdir, "logs")

        # new directory of the rotated input images
        self.NEW_DATA_DIR = "input_data" + os.sep

        # directory of the pretrained we
        self.PRETRAINED_MODEL = os.path.join(
            self.ROOT_DIR, "mask_rcnn_balloon.h5")

        # initialize default parameters
        self.VERBOSE = verbose
        self.ACTION = action
        self.FILE_NAME = fname
        self.FOLDER_NAME = fname
        self.OLD_DATA_DIR = fname
        self.NUM_INSTANCES = num_instances

        # accepted file extensions
        self.FILE_EXTENSION = (
            ".JPG", ".JPG".lower(),
            ".PNG", ".PNG".lower(),
            ".JPEG", ".JPEG".lower(),
            ".TIFF", ".TIFF".lower(),
        )

        self.RESULT_DIR = "results" + os.sep

        # location where masked apple trays will be saved
        self.FULLMASK_DIR = self.RESULT_DIR + "full_masked_images" + os.sep

        # location where segmented/individual instances will be saved
        self.SEGMENTED_DIR = self.RESULT_DIR + "segmented_images" + os.sep

        # location where apples with the scald removed will be saved
        self.BINARIZED_IMAGE = self.RESULT_DIR + "binarized_images" + os.sep

        # results for pear color bining
        self.BIN_COLOR = self.RESULT_DIR + "peel_color_results" + os.sep

    def check_path(self, dir, reset=0):
        """
                Check and create a directory if not exist

                Args: 
                        (str) dir: folder name
                        (int) reset: specify 1 to reset if the directory exists

                Returns: 
                        None
        """
        if os.path.exists(dir):
            if reset:
                shutil.rmtree(dir)
                os.makedirs(dir)
            pass
        else:
            os.makedirs(dir)

    def list_all(self, data_dir=os.path.curdir):
        """ 
                List all the folder names and image file names in the directory

                Args: 
                        (str) data_dir: the directory 

                Returns: 
                        (list) folder_name: all folders inside data_dir
                        (list) file_name: all files inside data_dir
        """
        file_name = []
        folder_name = []

        # if data_dir is a file
        if data_dir.endswith(self.FILE_EXTENSION):
            file_name.append(data_dir.split(os.sep)[-1])
            folder_name.append(data_dir.replace(data_dir.split(os.sep)[-1], os.path.curdir))
            return folder_name, file_name

        # list all folders and files in data_dir
        for root, dirs, files in os.walk(data_dir):

            # append the files to the list
            for file in files:
                if file.endswith(self.FILE_EXTENSION):
                    file_name.append(os.path.join(root, file))

            # append the folders to the list
            for fold in dirs:
                folder_name.append(os.path.join(root, fold))
            if folder_name == []:
                folder_name.append(os.path.join(root))

        return folder_name, file_name

    def clean_name(self, fname):
        """ 
                Remove file extensions in file names 

                Args: 
                        (str) fname: file names with extensions

                Returns: 
                        (str) fname: file names without file extensions
        """
        for ext in self.FILE_EXTENSION:
            fname = re.sub(ext, "", fname)
        return fname


class GrannyExtractInstances(GrannyBaseClass): 
    def __init__(self, action, fname, num_instances, verbose):
        super(GrannyExtractInstances, self).__init__(action, fname, num_instances, verbose)

    def load_model(self, verbose=1):
        """ 
                Load pretrained model, download if the model does not exist

                Args: 
                        (int) verbose: specify 0 to turn off model display

                Returns: 
                        (Keras model) model: model with the weights loaded
        """
        # download the pretrained weights from GitHub if not exist
        if not os.path.exists(self.PRETRAINED_MODEL):
            config.MRCNN_utils.download_trained_weights(self.PRETRAINED_MODEL)

        # load the configurations for the model
        AppleConfig = config.AppleConfig()
        if verbose:
            AppleConfig.display()

        # load model
        model = config.MRCNN_model.MaskRCNN(
            mode="inference", model_dir=self.MODEL_DIR, config=AppleConfig)

        # load pretrained weights to model
        model.load_weights(self.PRETRAINED_MODEL, by_name=True)
        return model

    def create_fullmask_image(self, model, im, fname=""):
        """ 
                Identify individual apples using the model 

                Args: 
                        (Keras model) model: Mask-RCNN model
                        (numpy.array) im: full image (tray of apples) to mask 
                        (str) data_dir: directory to save the masked image 
                        (str) fname: file (image) name

                Returns: 
                        (numpy.array) mask: [height, width, num_instances]
                        (numpy.array) box: [num_instance, (y1, x1, y2, x2, class_id)]
        """
        # detect image's instances using the model
        results = model.detect([im], verbose=0)
        r = results[0]

        # get the binary mask, box(coordinates), and confidence score from the result
        mask = r["masks"].astype(int)
        box = r["rois"]
        score = r["scores"]
        class_names = ["BG", ""]

        # display the image with the masks, box, and scores
        config.MRCNN_visualize.display_instances(im, box, mask, r['class_ids'],
                                                 class_names, score)

        # save the figure
        plt.savefig(os.path.join(fname + ".png"), bbox_inches='tight')

        return mask, box

    def label_instances_helper(self, df):
        """ 
                Helper function to sort the 18-apple tray using their center coordinates

                This sorting algorithm follows the numbering convention in 
                '01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'. 
                In an increasing order, sort by y-center coordinates then sort by x-center coordinates. 

                Args:
                (DataFrame) df: panda DataFrame containing coordinate information 

                Returns:
                (list) df_list: sorted coordinates of apples/pears
        """
        # sort df by y-center coordinates
        df = df.sort_values("ycenter", ascending=True, ignore_index=True)
        df.append(df.iloc[-1])

        # put the apples/pears into rows
        rows = 1
        count = 0
        for count in range(0, len(df)-1):
            df["rows"].iloc[count] = rows
            if not np.abs(df["ycenter"].iloc[count+1] - df["ycenter"].iloc[count]) < 300:
                rows += 1
        df["rows"].iloc[-1] = rows

        # sort apple/pear in each row using their x-center coordinates
        # if the first row has 5 apples/pears
        df_list = []
        if len(df[df["rows"] == 1]) == 5:
            apple_id = 18
            for i in range(1, 5):
                dfx = df[df["rows"] == i].sort_values(
                    "xcenter", ascending=False, inplace=False, ignore_index=True)
                for id in range(0, len(dfx)):
                    dfx["apple_id"].iloc[id] = apple_id
                    apple_id -= 1
                df_list.append(dfx)

        # if the first row has 4 apples/pears
        else:
            apple_id = 1
            for i in range(1, 5):
                dfx = df[df["rows"] == i].sort_values(
                    "xcenter", ascending=False, inplace=False, ignore_index=True)
                for id in range(0, len(dfx)):
                    dfx["apple_id"].iloc[id] = apple_id
                    apple_id += 1
                df_list.append(dfx)

        return df_list

    def sort_instances(self, box):
        """ 
                Sort and identify apples 
                This sorting algorithm follows the numbering convention in 
                '01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'

                Args:
                (numpy.array) box: [N, 4] where each row is y1, x1, y2, x2

                Returns:
                (numpy.array) apple_ar: sorted coordinates of apples/pears 
        """
        # convert to DataFrame
        df = pd.DataFrame(box)

        # label each column
        df.columns = ["y1", "x1", "y2", "x2"]

        # take first 18 rows (18 apples)
        df = df.iloc[0:18]

        # calculate centers for each apples
        df["ycenter"] = ((df["y1"]+df["y2"])/2).astype(int)
        df["xcenter"] = ((df["x1"]+df["x2"])/2).astype(int)

        # initialize columns
        df["rows"] = 0
        df["apple_id"] = 0
        df["nums"] = df.index

        # sort the DataFrame and return the list of instances
        apple_list = self.label_instances_helper(df)

        apple_ar = np.asarray(apple_list, dtype=object)
        return apple_ar

    def extract_image(self, sorted_arr, mask, im, fname=""):
        """ 
                Extract individual image from masks created by Mask-RCNN 

                Args: 
                        (numpy.array) sorted_arr: sorted coordinates of apples/pears
                        (numpy.array) mask: binary mask of individual apples
                        (numpy.array) im: full image (tray of apples) to extract 
                        (str) data_dir: directory to save the images
                        (str) fname: file name

                Returns: 
                        None
        """
        # loop over 18 apples/pears
        for k, ar in enumerate(sorted_arr):

            # loop over the coordinates
            for i in range(0, len(ar)):

                # make sure ar is np.array
                ar = np.array(ar)

                # take the corresponsing mask
                m = mask[:, :, ar[i][-1]]

                # initialize a blank array for the image
                new_im = np.zeros(
                    [ar[i][2]-ar[i][0], ar[i][3]-ar[i][1], 3], dtype=np.uint8)

                # extract individual image from the coordinates
                for j in range(0, im.shape[2]):
                    new_im[:, :, j] = im[ar[i][0]:ar[i][2], ar[i][1]:ar[i]
                                         [3], j]*m[ar[i][0]:ar[i][2], ar[i][1]:ar[i][3]]

                # save the image
                plt.imsave(fname + "_" + str(ar[i][-2]) + ".png", new_im)
    
    def rotate_image(self, old_im_dir, new_im_dir=""):
        """
                Check and rotate image 90 degree if needed to get 4000 x 6000

                Args: 
                        (str) old_im_dir: directory of the original image
                        (str) new_im_dir: directory of the rotated image

                Returns: 
                        (numpy.array) img: rotated image
        """
        img = skimage.io.imread(old_im_dir)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        skimage.io.imsave(new_im_dir, img)
        return img

    def mask_extract_image(self):
        """
                Main method performing Image Masking and Image Extraction on full tray images
                Output directory: 'segmented_data' and 'full_masked_data'
                        'segmented_data/': contains extracted images of individual apples
                        'full_masked_data/': contains masked images of apple trays
                Time: ~ 4-5 minutes per	 full-tray image	

                Args: 
                        None

                Returns: 
                        None
        """
        try:
            # load model
            model = self.load_model(verbose=self.VERBOSE)

            # list all folders and files
            data_dirs, file_names = self.list_all(self.OLD_DATA_DIR)

            # check and create a new "results" directory to store the results
            for data_dir in data_dirs:
                self.check_path(data_dir.replace(
                    self.OLD_DATA_DIR, self.FULLMASK_DIR))
                self.check_path(data_dir.replace(
                    self.OLD_DATA_DIR, self.SEGMENTED_DIR))
                self.check_path(data_dir.replace(
                    self.OLD_DATA_DIR, self.NEW_DATA_DIR))

            # pass each image to the model
            for file_name in file_names:
                name = file_name.split(os.sep)[-1]

                # print, for debugging purpose
                print(f"\t- Passing {name} into Mask R-CNN model. -")

                # check and rotate the image to landscape (4000x6000)
                img = self.rotate_image(
                    old_im_dir=file_name,
                    new_im_dir=file_name.replace(
                        self.OLD_DATA_DIR, self.NEW_DATA_DIR),
                )

                # remove file extension
                file_name = self.clean_name(file_name)

                # use the MRCNN model, identify individual apples/pear on trays
                mask, box = self.create_fullmask_image(
                    model=model,
                    im=img,
                    fname=file_name.replace(
                        self.OLD_DATA_DIR, self.FULLMASK_DIR)
                )

                # when NUM_INSTANCES = 18 (18 apples/pears) or NUM_INSTANCES not specified
                if self.NUM_INSTANCES == 1 or self.NUM_INSTANCES == 18:

                    # sort all instances using the convention in demo/18_apples_tray_convention.pdf
                    sorted_ar = self.sort_instances(box)

                    # extract the images
                    self.extract_image(sorted_arr=sorted_ar, mask=mask, im=img, fname=file_name.replace(
                        self.OLD_DATA_DIR, self.SEGMENTED_DIR))

                # when NUM_INSTANCES != 18
                else:

                    # the instances will not be sorted
                    warnings.warn(
                        "this is not a regular tray, the instances will not be sorted.")

                    # if there are more instances than NUM_INSTANCES
                    if self.NUM_INSTANCES > len(box):
                        print(
                            f"Only {len(box)} instances is detected.")
                        box = box

                    # if there are less instances than NUM_INSTANCES
                    else:
                        box = box[0:self.NUM_INSTANCES, :]

                    # concatenate the location array information
                    box = np.array(np.concatenate((box, np.array(
                        np.arange(1, len(box) + 1, dtype=int), ndmin=2).T, np.array(
                        np.arange(0, len(box), dtype=int), ndmin=2).T), axis=1))

                    # increase the dimensions to 2D
                    box = box[:, np.newaxis, :]

                    # extract the images
                    self.extract_image(sorted_arr=box, mask=mask, im=img, fname=file_name.replace(
                        self.OLD_DATA_DIR, self.SEGMENTED_DIR))

                # for debugging purpose
                print(
                    f"\t- {name} extracted. Check \"results/\" for output. - \n")
        except FileNotFoundError:
            print(f"\t- Folder/File Does Not Exist -")


class GrannySuperficialScald(GrannyBaseClass): 
    def __init__(self, action, fname, num_instances, verbose):
        super(GrannySuperficialScald, self).__init__(action, fname, num_instances, verbose)

    def remove_purple(self, img):
        """
                Remove the surrounding purple from the individual apples using YCrCb color space. 
                This function helps remove the unwanted regions for more precise calculation of the scald area. 


                Args: 
                (numpy.array) img: RGB, individual apple image 

                Returns: 
                (numpy.array) new_img: RGB, individual apple image with no purple regions 
        """
        # convert RGB to YCrCb
        new_img = img
        ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # create binary matrix (ones and zeros)
        bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

        # set max and min values for each channel
        channel1Min = 0*bin
        channel1Max = 255*bin
        channel2Min = 0*bin
        channel2Max = 255*bin
        channel3Min = 0*bin
        channel3Max = 126*bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater_equal(ycc_img[:, :, 0], channel1Min) & np.less_equal(
            ycc_img[:, :, 0], channel1Max)
        threshold_2 = np.greater_equal(ycc_img[:, :, 1], channel2Min) & np.less_equal(
            ycc_img[:, :, 1], channel2Max)
        threshold_3 = np.greater_equal(ycc_img[:, :, 2], channel3Min) & np.less_equal(
            ycc_img[:, :, 2], channel3Max)
        th123 = (threshold_1 & threshold_2 & threshold_3)

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def smooth_binary_mask(self, bin_mask):
        """
                Smooth scald region with basic morphological operations. 
                By performing morphology, the binary mask will be smoothened to avoid discontinuity. 

                Args: 
                        (numpy.array) bin_mask: binary mask (zeros & ones matrix) of the apples

                Returns: 
                        (numpy.array) bin_mask: smoothed binary mask (zeros & ones matrix) of the apples
        """
        bin_mask = np.uint8(bin_mask)

        # create a circular structuring element of size 10
        ksize = (10, 10)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        # using to structuring element to perform one close and one open operation on the binary mask
        bin_mask = cv2.dilate(
            cv2.erode(bin_mask, kernel=strel, iterations=1), kernel=strel, iterations=1)
        bin_mask = cv2.erode(cv2.dilate(
            bin_mask, kernel=strel, iterations=1), kernel=strel, iterations=1)
        return bin_mask

    def remove_scald(self, img):
        """
                Remove the scald region from the individual apple images. 
                Note that the stem could have potentially been removed during the process. 

                Args: 
                        (numpy.array) img: RGB, individual apple image 

                Returns: 
                        (numpy.array) new_img: RGB, individual apple image with the scald region removed
                        (numpy.array) th123: binary mask 
        """
        # convert from RGB to Lab color space
        new_img = img
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)

        # rescale all 3 channels to the 0-90 range
        for i in range(3):
            lab_img[:, :, i] = (lab_img[:, :, i] - np.amin(lab_img[:, :, i])) / \
                (np.amax(lab_img[:, :, i]) - np.amin(lab_img[:, :, i]))
            lab_img[:, :, i] *= 90

        # get channel 2 histogram for min and max values
        lab_img = lab_img.astype(np.uint8)
        hist, bin_edges = np.histogram(lab_img[:, :, 1], bins=90)

        # create binary matrix (ones and zeros)
        bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

        # set max and min values for each channel
        im_range = np.float32(90)
        channel1Min = 1*bin
        channel1Max = im_range*bin
        channel2Min = -1/2*im_range*bin
        channel2Max = (np.argmax(hist) - 3/9*im_range + 1)*bin
        channel3Min = 1*bin
        channel3Max = im_range*bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater_equal(lab_img[:, :, 0], channel1Min) & np.less_equal(
            lab_img[:, :, 0], channel1Max)
        threshold_2 = np.greater_equal(lab_img[:, :, 1], channel2Min) & np.less_equal(
            lab_img[:, :, 1], channel2Max)
        threshold_3 = np.greater_equal(lab_img[:, :, 2], channel3Min) & np.less_equal(
            lab_img[:, :, 2], channel3Max)
        th123 = (threshold_1 & threshold_2 & threshold_3)

        # perform simple morphological operation to smooth the binary mask
        th123 = self.smooth_binary_mask(th123)

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return th123, new_img
    
    def score_image(self, img):
        """ 
                Clean up individual image (remove purple area of the tray),
                and remove scald 

                Args: 
                        (numpy.array) img: individual apple/pear image to rate

                Returns: 
                        (numpy.array) nopurple_img: 
                        (numpy.array) img: RGB, no scald image 
                        (numpy.array) bw: binary mask (zeros & ones array)
        """

        # Remove surrounding purple
        img = self.remove_purple(img)
        nopurple_img = img

        # Image smoothing
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

        # Image binarization (Removing scald regions)
        bw, img = self.remove_scald(img)

        return nopurple_img, img, bw

    def calculate_scald(self, bw, img):
        """
                Calculate scald region by counting all non zeros area

                Args: 
                        (numpy.array) bw: binarized image
                        (numpy.array) img: original image to be used as ground truth

                Returns: 
                        (float) fraction: the scald region, i.e. fraction of the original image that was removed
        """
        # convert to uint8
        img = np.uint8(img)

        # count non zeros of binarized image
        ground_area = 1/3*np.count_nonzero(img[:, :, 0:2])

        # count non zeros of original image
        mask_area = 1/3*np.count_nonzero(bw[:, :, 0:2])

        # calculate fraction
        fraction = 0
        if ground_area == 0:
            return 1
        else:
            fraction = 1 - mask_area/ground_area

        if fraction < 0:
            return 0
        return fraction
    
    def rate_binarize_image(self):
        """ 
                (GS) Main method performing Image Binarization, i.e. rate and remove scald, on individual apple images 
                
                This is the main method being called by the Python argument parser from the command.py to set up CLI for 
                "Granny Smith" superficial scald calculation. 
                The calculated scores will be written to a .csv file.

                Args: 
                        None

                Returns: 
                        None
        """
        # create "results" directory to save the results
        self.check_path(self.BINARIZED_IMAGE)

        # single-image rating
        if self.NUM_INSTANCES == 1:
            try:
                # read the image from file
                file_name = self.FILE_NAME
                img = skimage.io.imread(file_name)

                # remove the surroundings
                nopurple_img, binarized_image, bw = self.score_image(img)

                # calculate the scald region and save image
                score = self.calculate_scald(binarized_image, nopurple_img)
                file_name = file_name.split(os.sep)[-1]
                skimage.io.imsave(os.path.join(
                    self.BINARIZED_IMAGE, file_name), binarized_image)

                # save the scores to results/rating.csv
                with open("results" + os.sep + "scald_ratings.csv", "w") as w:
                    w.writelines(f"{self.clean_name(file_name)}:\t\t{score}")
                    w.writelines("\n")
                print(f"\t- Done. Check \"results/\" for output. - \n")
            except FileNotFoundError:
                print(f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -")

        # multi-images rating
        else:
            try:
                # list all files and folders in the folder
                folders, files = self.list_all(self.FOLDER_NAME)

                # create "results" directory to save the results
                for folder in folders:
                    self.check_path(folder.replace(
                        self.FOLDER_NAME, self.BINARIZED_IMAGE))

                # remove scald and rate each apple
                scores = []
                for file_name in files:

                    # read the image from file
                    img = skimage.io.imread(file_name)
                    file_name = self.clean_name(file_name)

                    # remove the surroundings
                    nopurple_img, binarized_image, bw = self.score_image(img)

                    # calculate the scald region and save image
                    score = self.calculate_scald(binarized_image, nopurple_img)
                    file_name = file_name.split(os.sep)[-1]
                    scores.append(score)
                    skimage.io.imsave(os.path.join(
                        self.BINARIZED_IMAGE, file_name + ".png"), binarized_image)

                # save the scores to results/rating.csv
                with open("results" + os.sep + "scald_ratings.csv", "w") as w:
                    for i, score in enumerate(scores):
                        w.writelines(
                            f"{self.clean_name(files[i])}:\t\t{score}")
                        w.writelines("\n")
                    print(f"\t- Done. Check \"results/\" for output. - \n")
            except FileNotFoundError:
                print(f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values.-")


class GrannyPeelColor(GrannyBaseClass): 
    def __init__(self, action, fname, num_instances, verbose): 
        super(GrannyPeelColor, self).__init__(action, fname, num_instances, verbose)
        self.MEAN_VALUES_L = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        self.MEAN_VALUES_A = [
            -22.73535185450301,
            -35.856673734593976,
            -14.839191848288477,
            -38.33945889256972,
            -38.35401900977177,
            -25.183705217005663,
            -37.46455193845067,
            -18.57975296251061,
            -20.153556829155015,
            -28.66620337913712,
            -15.422998269835011
        ]
        self.MEAN_VALUES_B = [
            84.36038598038662,
            72.21634388774689,
            91.21422051166374,
            60.41978876347979,
            60.40667779362334,
            81.7513027496333,
            58.068541555881474,
            86.95338287223824,
            78.30049344632643,
            77.42875637928557,
            92.37148315325989
        ]

    def remove_purple(self, img):
        """
                Remove the surrounding purple from the individual apples using YCrCb color space. 
                This function helps remove the unwanted regions for more precise calculation of the scald area. 

                Args: 
                (numpy.array) img: RGB, individual apple image 

                Returns: 
                (numpy.array) new_img: RGB, individual apple image with no purple regions 
        """
        # convert RGB to YCrCb
        new_img = img
        ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # create binary matrix (ones and zeros)
        bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

        # set max and min values for each channel
        channel1Min = 0*bin
        channel1Max = 255*bin
        channel2Min = 0*bin
        channel2Max = 255*bin
        channel3Min = 0*bin
        channel3Max = 126*bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater_equal(ycc_img[:, :, 0], channel1Min) & np.less_equal(
            ycc_img[:, :, 0], channel1Max)
        threshold_2 = np.greater_equal(ycc_img[:, :, 1], channel2Min) & np.less_equal(
            ycc_img[:, :, 1], channel2Max)
        threshold_3 = np.greater_equal(ycc_img[:, :, 2], channel3Min) & np.less_equal(
            ycc_img[:, :, 2], channel3Max)
        th123 = (threshold_1 & threshold_2 & threshold_3)

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img


    def get_green_yellow_values(self, img): 
        """
                Get the mean pixel values from the images representing the amount of 
                green and yellow in the CIELAB color space. Then, normalize the values to L = 50.

                Args: 
                        (numpy.array) img: RGB, individual apple image 

                Returns: 
                        (numpy.float) scaled_l: normalized mean values of L channel 
                        (numpy.float) scaled_a: normalized mean values of a channel 
                        (numpy.float) scaled_b: normalized mean values of b channel 
        """
        # convert from RGB to Lab color space
        new_img = img
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)

        # get channel 2 histogram for min and max values
        lab_img = lab_img.astype(np.uint8)

        # create binary matrix (ones and zeros)
        bin = np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0)

        # set max and min values for each channel
        channel1Min = 0*bin
        channel1Max = 255*bin
        channel2Min = 0*bin
        channel2Max = 128*bin
        channel3Min = 128*bin
        channel3Max = 255*bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater(lab_img[:, :, 0], channel1Min) & np.less(
            lab_img[:, :, 0], channel1Max)
        threshold_2 = np.greater(lab_img[:, :, 1], channel2Min) & np.less(
            lab_img[:, :, 1], channel2Max)
        threshold_3 = np.greater(lab_img[:, :, 2], channel3Min) & np.less(
            lab_img[:, :, 2], channel3Max)
        th123 = (threshold_1 & threshold_2 & threshold_3)

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        
        # get mean values from each channel 
        mean_l = np.sum(lab_img[:,:,0])/np.count_nonzero(threshold_1) * 100/255
        mean_a = np.sum(lab_img[:,:,1]*threshold_2)/np.count_nonzero(threshold_2) - 128
        mean_b = np.sum(lab_img[:,:,2]*threshold_3)/np.count_nonzero(threshold_3) - 128

        # normalize by shifting point in the spherical coordinates
        radius = np.sqrt(mean_l**2 + mean_a**2 + mean_b**2)
        scaled_l = 50
        scaled_a = np.sign(mean_a)*np.sqrt((radius**2 - scaled_l**2)/(1 + (mean_b/mean_a)**2))
        scaled_b = np.sign(mean_b)*mean_b/mean_a*scaled_a

        return scaled_l, scaled_a, scaled_b
    
    def calculate_bin_distance(self, color_list, method = "Euclidean"): 
        """
            Calculate the Euclidean distance from normalized image's LAB to each bin color. 
            Return the shortest distance and the corresponding bin. 

            Args: 
                    (list) color_list: 1x3 of [mean pixel L, mean pixel a, mean pixel b]

            Returns: 
                    
        """
        bin_num = 0 
        dist = 0
        if method == "Euclidean": 
            dist = np.sqrt((color_list[1] - np.array(self.MEAN_VALUES_A))**2, (color_list[2] - np.array(self.MEAN_VALUES_B))**2) 
            bin_num = np.argmin(dist)
        
        return bin_num, dist
    
    def sort_peel_color(self): 
        # create "results" directory to save the results
        self.check_path(self.BIN_COLOR)
        if self.NUM_INSTANCES == 1: 
            try:
                # create "results" directory to save the results
                self.check_path(self.RESULT_DIR)

                # read image 
                file_name = self.FILE_NAME
                img = skimage.io.imread(file_name)

                # remove surrounding purple
                img = self.remove_purple(img)
                nopurple_img = img 

                # image smoothing
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

                # get image values
                l, a, b = self.get_green_yellow_values(img)

                # calculate distance to each bin 
                bin_num, distances = self.calculate_bin_distance([l, a, b])

                # convert to string
                string_dist = ""
                for dist in distances: 
                    string_dist += dist + ","

                # save the scores to results/rating.csv
                with open(self.BIN_COLOR + os.sep + "peel_colors.csv", "w") as w:
                    w.writelines(f"{self.clean_name(file_name)},{bin_num},{string_dist}")
                    w.writelines("\n")
                print(f"\t- Done. Check \"results/\" for output. - \n")

            except:
                print(f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -")
        
        else: 
            try: 
                # list all files and folders in the folder
                folders, files = self.list_all(self.FOLDER_NAME)

                # create "results" directory to save the results
                for folder in folders:
                    self.check_path(folder.replace(
                        self.FOLDER_NAME, self.BIN_COLOR))
                
                bin_nums = []
                distances = []
                for file_name in files: 
                    img = skimage.io.imread(file_name)

                    # remove surrounding purple
                    img = self.remove_purple(img)
                    nopurple_img = img 

                    # image smoothing
                    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

                    # get image values
                    l, a, b = self.get_green_yellow_values(img)

                    # calculate distance to each bin 
                    bin_num, distance = self.calculate_bin_distance([l, a, b])

                    # convert to string
                    string_dist = ""
                    for dist in distance: 
                        string_dist += dist + ","
                    
                    bin_nums.append(bin_num)
                    distances.append(string_dist)

                with open(self.BIN_COLOR + os.sep + "peel_colors.csv", "w") as w: 
                    for i in len(bin_nums): 
                        w.writelines(f"{self.clean_name(files[i])},{bin_nums[i]},{string_dist[i]}")
                        w.writelines("\n")
                    print(f"\t- Done. Check \"results/\" for output. - \n")
            except: 
                print(f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -")


class GrannyStarchIndex(GrannyBaseClass): 
    def __init__(self, action, fname, num_instances, verbose): 
        super(GrannyBaseClass).__init__(self, action, fname, num_instances, verbose)
    
    def main(): 
        pass
