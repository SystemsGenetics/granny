import os
from multiprocessing import Pool
from typing import List, Tuple

import cv2
import numpy as np
from GRANNY import GRANNY_Base as granny
from numpy.typing import NDArray


class GrannySuperficialScald(granny.GrannyBase):
    def __init__(self, action: str, fname: str, num_instances: int):
        num_instances = 1 if num_instances == None else num_instances
        super(GrannySuperficialScald, self).__init__(action, fname, num_instances)

    def remove_purple(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Remove the surrounding purple from the individual apples using YCrCb color space.
        This function helps remove the unwanted regions for more precise calculation of the scald area.
        """
        # convert RGB to YCrCb
        new_img = img.copy()
        ycc_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        # create binary matrices
        threshold_1 = np.logical_and((ycc_img[:, :, 0] >= 0), (ycc_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and((ycc_img[:, :, 1] >= 0), (ycc_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and((ycc_img[:, :, 2] >= 0), (ycc_img[:, :, 2] <= 126))

        # combine to one matrix
        th123 = np.logical_and(np.logical_and(threshold_1, threshold_2), threshold_3).astype(
            np.uint8
        )

        # create new image using threshold matrices
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123
        return new_img

    def smooth_binary_mask(self, bin_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Smooth scald region with basic morphological operations.
        By performing morphology, the binary mask will be smoothened to avoid discontinuity.
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

    def remove_scald(self, img: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """
        Remove the scald region from the individual apple images.
        Note that the stem could have potentially been removed during the process.
        """
        # convert from RGB to Lab color space
        new_img = img.copy()
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

        def calculate_threshold_from_hist(hist: NDArray[np.int8]) -> int:
            hist_range = 255 - (hist[::-1] != 0).argmax() - (hist != 0).argmax()
            threshold = np.max(np.argsort(hist)[-10:])
            threshold = int(threshold - 1 / 3 * hist_range)
            threshold = 100 if threshold < 100 else int(threshold)
            return threshold

        # create binary matrices
        hist, _ = np.histogram(lab_img[:, :, 1], bins=256, range=(0, 255))
        threshold = calculate_threshold_from_hist(hist)
        threshold_1 = np.logical_and((lab_img[:, :, 0] >= 1), (lab_img[:, :, 0] <= 255))
        threshold_2 = np.logical_and((lab_img[:, :, 1] >= 1), (lab_img[:, :, 1] <= threshold))
        threshold_3 = np.logical_and((lab_img[:, :, 2] >= 1), (lab_img[:, :, 2] <= 255))

        # combine to one matrix
        th123 = np.logical_and(np.logical_and(threshold_1, threshold_2), threshold_3).astype(
            np.uint8
        )

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
        nopurple_img = img.copy()

        # Image smoothing
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

        # Image binarization (Removing scald regions)
        bw, img = self.remove_scald(img)

        return nopurple_img, img, bw

    def calculate_scald(self, bw: NDArray[np.uint8], img: NDArray[np.uint8]) -> float:
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

    def rate_GrannySmith_superficial_scald(self, file_name: str) -> float:
        img = cv2.cvtColor(cv2.imread(file_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        print(f"\t- Rating {file_name}. -")

        # remove the surroundings
        nopurple_img, binarized_image, bw = self.score_image(img)

        # calculate the scald region and save image
        score = self.calculate_scald(binarized_image, nopurple_img)

        print(f"\t- Score: {score}. -")
        cv2.imwrite(
            os.path.join(self.BINARIZED_IMAGE, os.path.basename(file_name)),
            cv2.cvtColor(binarized_image, cv2.COLOR_RGB2BGR),
        )
        return score

    def rate_GrannySmith_superficial_scald_multiprocessing(self, args: str) -> float:
        """Rates GrannySmith superficial scald using multiprocessing library"""
        file_name = args
        score = self.rate_GrannySmith_superficial_scald(os.path.join(self.FOLDER_NAME, file_name))
        return score

    def GrannySuperficialScald(self) -> None:
        self.create_directories(self.RESULT_DIR, self.BINARIZED_IMAGE)
        image_list = os.listdir(self.FOLDER_NAME)
        cpu_count = int(os.cpu_count()*0.8) or 1
        with Pool(cpu_count) as pool:
            results = pool.map(self.rate_GrannySmith_superficial_scald_multiprocessing, image_list)

        image_list = sorted(image_list)
        with open(f"results{os.sep}scald_ratings.csv", "w") as w:
            for i, file_name in enumerate(image_list):
                w.writelines(f"{self.FOLDER_NAME}/{file_name}\t\t{results[i]}")
                w.writelines("\n")
            print(f'\t- Done. Check "results/" for output. - \n')

        # with Pool(cpu_count) as pool:
        #     for _ in tqdm(
        #         pool.imap_unordered(
        #             self.rate_GrannySmith_superficial_scald_multiprocessing,
        #             [image for image in image_list],
        #         ),
        #         total=len(image_list),
        #     ):
        #         pass

    # def rate_GrannySmith_superficial_scald(self) -> None:
    #     """
    #     (GS) Main method performing Image Binarization, i.e. rate and remove scald, on individual apple images

    #     This is the main method being called by the Python argument parser from the command.py to set up CLI for
    #     "Granny Smith" superficial scald calculation. The calculated scores will be written to a .csv file.
    #     """
    #     # create "results" directory to save the results
    #     self.create_directories(self.BINARIZED_IMAGE)

    #     # single-image rating
    #     if self.NUM_INSTANCES == 1:
    #         # read the image from file
    #         file_name = self.FILE_NAME

    # img = skimage.io.imread(file_name)

    # print(f"\t- Rating {file_name}. -")

    # # remove the surroundings
    # nopurple_img, binarized_image, _ = self.score_image(img)

    # # calculate the scald region and save image
    # score = self.calculate_scald(binarized_image, nopurple_img)

    # print(f"\t- Score: {score}. -")
    # file_name = file_name.split(os.sep)[-1]
    # skimage.io.imsave(
    #     os.path.join(self.BINARIZED_IMAGE, file_name),
    #     binarized_image,
    # )

    #         # save the scores to results/rating.csv
    #         with open("results" + os.sep + "scald_ratings.csv", "w") as w:
    #             w.writelines(f"{self.clean_name(file_name)}:\t\t{score}")
    #             w.writelines("\n")
    #         print(f'\t- Done. Check "results/" for output. - \n')

    #     # multi-images rating
    #     else:
    #         # list all files and folders in the folder
    #         folders, files = self.list_all(self.FOLDER_NAME)

    #         # create "results" directory to save the results
    #         for folder in folders:
    #             self.create_directories(folder.replace(self.FOLDER_NAME, self.BINARIZED_IMAGE))

    #         # remove scald and rate each apple
    #         scores: List[float] = []
    #         for file_name in files:
    #             print(f"\t- Rating {file_name}. -")

    #             # read the image from file
    #             img = skimage.io.imread(file_name)
    #             file_name = self.clean_name(file_name)

    #             # remove the surroundings
    #             nopurple_img, binarized_image, _ = self.score_image(img)

    #             # calculate the scald region and save image
    #             score = self.calculate_scald(binarized_image, nopurple_img)
    #             file_name = file_name.split(os.sep)[-1]
    #             scores.append(score)
    #             skimage.io.imsave(
    #                 os.path.join(self.BINARIZED_IMAGE, file_name + ".png"),
    #                 binarized_image,
    #             )

    #         # save the scores to results/rating.csv
    #         with open(f"results{os.sep}scald_ratings.csv", "w") as w:
    #             for i, score in enumerate(scores):
    #                 w.writelines(f"{self.clean_name(files[i])}:\t\t{score}")
    #                 w.writelines("\n")
    #             print(f'\t- Done. Check "results/" for output. - \n')
