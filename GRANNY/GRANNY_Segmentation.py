import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from GRANNY import GRANNY_Base as granny
from GRANNY import GRANNY_config as config
from Mask_RCNN.model import MaskRCNN
from numpy.typing import NDArray


class GrannySegmentation(granny.GrannyBase):
    def __init__(self, action: str, fname: str, num_instances: int):
        num_instances = 18 if num_instances == None else num_instances
        super(GrannySegmentation, self).__init__(action, fname, num_instances)

    def load_model(self, verbose: int = 0) -> MaskRCNN:
        """
        Load pretrained model, download if the model does not exist
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
            mode="inference", model_dir=self.MODEL_DIR, config=AppleConfig
        )

        # load pretrained weights to model
        model.load_weights(self.PRETRAINED_MODEL, by_name=True)
        return model

    def create_fullmask_image(
        self, model: MaskRCNN, im: NDArray[np.uint8], fname: str
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Identify individual apples using the model
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
        config.MRCNN_visualize.display_instances(im, box, mask, r["class_ids"], class_names, score)

        # save the figure
        plt.savefig(os.path.join(fname + ".png"), bbox_inches="tight")

        return mask, box

    def label_instances_helper(self, df: pd.DataFrame):
        """
        Helper function to sort the 18-apple tray using their center coordinates

        This sorting algorithm follows the numbering convention in
        '01-input_data/GS-1-16_FilesForImageAnalysis/GS-1-16_ImageTutorial.pptx'.
        In an increasing order, sort by y-center coordinates then sort by x-center coordinates.
        """
        # sort df by y-center coordinates
        df = df.sort_values("ycenter", ascending=True, ignore_index=True)
        df.append(df.iloc[-1])

        # put the apples/pears into rows
        rows = 1
        count = 0
        for count in range(0, len(df) - 1):
            df["rows"].iloc[count] = rows
            if not np.abs(df["ycenter"].iloc[count + 1] - df["ycenter"].iloc[count]) < 300:
                rows += 1
        df["rows"].iloc[-1] = rows

        # sort apple/pear in each row using their x-center coordinates
        # if the first row has 5 apples/pears
        df_list = []
        if len(df[df["rows"] == 1]) == 5:
            apple_id = self.NUM_INSTANCES
            for i in range(1, 5):
                dfx = df[df["rows"] == i].sort_values(
                    "xcenter",
                    ascending=False,
                    inplace=False,
                    ignore_index=True,
                )
                for id in range(0, len(dfx)):
                    dfx["apple_id"].iloc[id] = apple_id
                    apple_id -= 1
                df_list.append(dfx)

        # if the first row has 4 apples/pears
        else:
            apple_id = 1
            for i in range(1, 5):
                dfx = df[df["rows"] == i].sort_values(
                    "xcenter",
                    ascending=False,
                    inplace=False,
                    ignore_index=True,
                )
                for id in range(0, len(dfx)):
                    dfx["apple_id"].iloc[id] = apple_id
                    apple_id += 1
                df_list.append(dfx)

        return df_list

    def sort_instances(self, box: NDArray[np.uint8]):
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
        df = df.iloc[0 : self.NUM_INSTANCES]

        # calculate centers for each apples
        df["ycenter"] = ((df["y1"] + df["y2"]) / 2).astype(int)
        df["xcenter"] = ((df["x1"] + df["x2"]) / 2).astype(int)

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
                    [ar[i][2] - ar[i][0], ar[i][3] - ar[i][1], 3],
                    dtype=np.uint8,
                )

                # extract individual image from the coordinates
                for j in range(0, im.shape[2]):
                    new_im[:, :, j] = (
                        im[ar[i][0] : ar[i][2], ar[i][1] : ar[i][3], j]
                        * m[ar[i][0] : ar[i][2], ar[i][1] : ar[i][3]]
                    )

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
        if self.ACTION == "pear":
            img = cv2.flip(img, 1)
        skimage.io.imsave(new_im_dir, img)
        return img

    def extract_instances_with_MaskRCNN(self):
        """
        Main method performing Image Masking and Image Extraction on full tray images
        Output directory: 'segmented_data' and 'full_masked_data'
                'segmented_data/': contains extracted images of individual apples
                'full_masked_data/': contains masked images of apple trays
        Time: ~ 4-5 minutes per 4000 x 6000	full-tray image
        """
        try:
            # load model
            view_architecture = 0
            model = self.load_model(verbose=view_architecture)

            # list all folders and files
            data_dirs, file_names = self.list_all(self.INPUT_FNAME)

            # check and create a new "results" directory to store the results
            for data_dir in data_dirs:
                self.create_directories(data_dir.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR))
                self.create_directories(data_dir.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR))
                self.create_directories(data_dir.replace(self.OLD_DATA_DIR, self.NEW_DATA_DIR))

            # pass each image to the model
            for file_name in file_names:
                name = file_name.split(os.sep)[-1]

                # print, for debugging purpose
                print(f"\t- Passing {name} into Mask R-CNN model. -")

                # check and rotate the image to landscape (4000x6000)
                img = self.rotate_image(
                    old_im_dir=file_name,
                    new_im_dir=file_name.replace(self.OLD_DATA_DIR, self.NEW_DATA_DIR),
                )

                # remove file extension
                file_name = self.clean_name(file_name)

                # use the MRCNN model, identify individual apples/pear on trays
                mask, box = self.create_fullmask_image(
                    model=model,
                    im=img,
                    fname=file_name.replace(self.OLD_DATA_DIR, self.FULLMASK_DIR),
                )

                # if there are more instances than NUM_INSTANCES
                if self.NUM_INSTANCES > len(box):
                    print(f"Only {len(box)} instances is detected.")
                    box = box

                # if there are less instances than NUM_INSTANCES
                else:
                    box = box[0 : self.NUM_INSTANCES, :]

                # sort all instances using the convention in demo/18_apples_tray_convention.pdf
                sorted_ar = self.sort_instances(box)

                # extract the images
                self.extract_image(
                    sorted_arr=sorted_ar,
                    mask=mask,
                    im=img,
                    fname=file_name.replace(self.OLD_DATA_DIR, self.SEGMENTED_DIR),
                )

                # for debugging purpose
                print(f'\t- {name} done. Check "results/" for output. - \n')
        except FileNotFoundError:
            print(f"\t- Folder/File Does Not Exist -")
