import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import skimage
from GRANNY import GRANNY_Base as granny
from numpy.typing import NDArray


class GrannySuperficialScald(granny.GrannyBase):
    def __init__(self, action: str, fname: str, num_instances: int, verbose):
        num_instances = 1 if num_instances == None else num_instances
        verbose = 0 if verbose == None else 1
        super(GrannySuperficialScald, self).__init__(
            action, fname, num_instances, verbose
        )

    def remove_purple(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Remove the surrounding purple from the individual apples using YCrCb color space.
        This function helps remove the unwanted regions for more precise calculation of the scald area.
        """
        # convert RGB to YCrCb
        new_img = img
        ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # create binary matrix (ones and zeros)
        bin = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)

        # set max and min values for each channel
        channel1Min = 0 * bin
        channel1Max = 255 * bin
        channel2Min = 0 * bin
        channel2Max = 255 * bin
        channel3Min = 0 * bin
        channel3Max = 126 * bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater_equal(
            ycc_img[:, :, 0], channel1Min
        ) & np.less_equal(ycc_img[:, :, 0], channel1Max)
        threshold_2 = np.greater_equal(
            ycc_img[:, :, 1], channel2Min
        ) & np.less_equal(ycc_img[:, :, 1], channel2Max)
        threshold_3 = np.greater_equal(
            ycc_img[:, :, 2], channel3Min
        ) & np.less_equal(ycc_img[:, :, 2], channel3Max)
        th123 = threshold_1 & threshold_2 & threshold_3

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def smooth_binary_mask(
        self, bin_mask: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Smooth scald region with basic morphological operations.
        By performing morphology, the binary mask will be smoothened to avoid discontinuity.

        Args:
                (numpy.array) bin_mask: binary mask (zeros & ones matrix) of the apples

        Returns:
                (numpy.array) bin_mask: smoothed binary mask (zeros & ones matrix) of the apples
        """
        bin_mask = bin_mask

        # create a circular structuring element of size 10
        ksize = (10, 10)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        # using to structuring element to perform one close and one open operation on the binary mask
        bin_mask = cv2.dilate(
            cv2.erode(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )
        bin_mask = cv2.erode(
            cv2.dilate(bin_mask, kernel=strel, iterations=1),
            kernel=strel,
            iterations=1,
        )
        return bin_mask

    def remove_scald(
        self, img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Remove the scald region from the individual apple images.
        Note that the stem could have potentially been removed during the process.
        """
        # convert from RGB to Lab color space
        new_img: NDArray[np.uint8] = img
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # create binary matrix (ones and zeros)
        bin = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)
        hist, bin_edges = np.histogram(
            lab_img[:, :, 1], bins=256, range=(0, 255)
        )
        hist_range = 255 - (hist[::-1] != 0).argmax() - (hist != 0).argmax()
        threshold = np.max(np.argsort(hist)[-10:])
        threshold = int(threshold - 2 / 3 * hist_range)
        threshold = 100 if threshold < 100 else int(threshold)

        # set max and min values for each channel
        channel1Min = 1 * bin
        channel1Max = 255 * bin
        channel2Min = 1 * bin
        channel2Max = threshold * bin
        channel3Min = 1 * bin
        channel3Max = 255 * bin
        # print(f"Anticipated score: {1 - np.count_nonzero(lab_img[:,:,1]<=threshold)/np.count_nonzero(img[:,:,0]!=0)}")

        # create threshold matrices for each for each channel
        threshold_1 = np.greater_equal(
            lab_img[:, :, 0], channel1Min
        ) & np.less_equal(lab_img[:, :, 0], channel1Max)
        threshold_2 = np.greater_equal(
            lab_img[:, :, 1], channel2Min
        ) & np.less_equal(lab_img[:, :, 1], channel2Max)
        threshold_3 = np.greater_equal(
            lab_img[:, :, 2], channel3Min
        ) & np.less_equal(lab_img[:, :, 2], channel3Max)
        th123 = threshold_1 & threshold_2 & threshold_3

        # perform simple morphological operation to smooth the binary mask
        th123 = self.smooth_binary_mask(th123)

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return th123, new_img

    def score_image(
        self, img: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Clean up individual image (remove purple area of the tray), and remove scald
        """

        # Remove surrounding purple
        img = self.remove_purple(img)
        nopurple_img = img

        # Image smoothing
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

        # Image binarization (Removing scald regions)
        bw, img = self.remove_scald(img)

        return nopurple_img, img, bw

    def calculate_scald(
        self, bw: NDArray[np.uint8], img: NDArray[np.uint8]
    ) -> float:
        """
        Calculate scald region by counting all non zeros area

        Args:
                (numpy.array) bw: binarized image
                (numpy.array) img: original image to be used as ground truth

        Returns:
                (float) fraction: the scald region, i.e. fraction of the original image that was removed
        """
        # convert to uint8
        img = img

        # count non zeros of binarized image
        ground_area = 1 / 3 * np.count_nonzero(img[:, :, 0:2])

        # count non zeros of original image
        mask_area = 1 / 3 * np.count_nonzero(bw[:, :, 0:2])

        # calculate fraction
        fraction = 0
        if ground_area == 0:
            return 1
        else:
            fraction = 1 - mask_area / ground_area
        if fraction < 0:
            return 0
        return fraction

    def rate_GrannySmith_superficial_scald(self):
        """
        (GS) Main method performing Image Binarization, i.e. rate and remove scald, on individual apple images

        This is the main method being called by the Python argument parser from the command.py to set up CLI for
        "Granny Smith" superficial scald calculation. The calculated scores will be written to a .csv file.
        """
        # create "results" directory to save the results
        self.create_directories(self.BINARIZED_IMAGE)

        # single-image rating
        if self.NUM_INSTANCES == 1:
            try:
                # read the image from file
                file_name = self.FILE_NAME

                img = skimage.io.imread(file_name)

                print(f"\t- Rating {file_name}. -")

                # remove the surroundings
                nopurple_img, binarized_image, bw = self.score_image(img)

                # calculate the scald region and save image
                score = self.calculate_scald(binarized_image, nopurple_img)

                print(f"\t- Score: {score}. -")
                file_name = file_name.split(os.sep)[-1]
                skimage.io.imsave(
                    os.path.join(self.BINARIZED_IMAGE, file_name),
                    binarized_image,
                )

                # save the scores to results/rating.csv
                with open("results" + os.sep + "scald_ratings.csv", "w") as w:
                    w.writelines(f"{self.clean_name(file_name)}:\t\t{score}")
                    w.writelines("\n")
                print(f'\t- Done. Check "results/" for output. - \n')
            except FileNotFoundError:
                print(
                    f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -"
                )

        # multi-images rating
        else:
            try:
                # list all files and folders in the folder
                folders, files = self.list_all(self.FOLDER_NAME)

                # create "results" directory to save the results
                for folder in folders:
                    self.create_directories(
                        folder.replace(self.FOLDER_NAME, self.BINARIZED_IMAGE)
                    )

                # remove scald and rate each apple
                scores: List[float] = []
                for file_name in files:
                    print(f"\t- Rating {file_name}. -")

                    # read the image from file
                    img = skimage.io.imread(file_name)
                    file_name = self.clean_name(file_name)

                    # remove the surroundings
                    nopurple_img, binarized_image, bw = self.score_image(img)

                    # calculate the scald region and save image
                    score = self.calculate_scald(binarized_image, nopurple_img)
                    file_name = file_name.split(os.sep)[-1]
                    scores.append(score)
                    skimage.io.imsave(
                        os.path.join(self.BINARIZED_IMAGE, file_name + ".png"),
                        binarized_image,
                    )

                # save the scores to results/rating.csv
                with open("results" + os.sep + "scald_ratings.csv", "w") as w:
                    for i, score in enumerate(scores):
                        w.writelines(
                            f"{self.clean_name(files[i])}:\t\t{score}"
                        )
                        w.writelines("\n")
                    print(f'\t- Done. Check "results/" for output. - \n')
            except FileNotFoundError:
                print(
                    f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values.-"
                )
