import os
from multiprocessing import Pool
from typing import Any, cast

import cv2
import numpy as np
from GRANNY import GRANNY_Base as granny
from numpy.typing import NDArray


class GrannyStarchArea(granny.GrannyBase):
    def __init__(self, action: str, fname: str):
        super(GrannyStarchArea, self).__init__(action, fname)

    def calculate_starch(self, file_name: str) -> float:
        # Load the image
        img = cast(NDArray[np.uint8], cv2.imread(file_name, cv2.IMREAD_COLOR))
        new_img = img.copy()
        img = cv2.GaussianBlur(img, (11, 11), 0)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        def calculate_threshold_from_hist(hist: NDArray[np.int8]) -> int:
            histogram_sum = np.sum(hist)
            left_sum = 0

            for i, bin_value in enumerate(hist):
                left_sum += bin_value
                right_sum = histogram_sum - left_sum

                if left_sum - right_sum > 0.1 * histogram_sum:
                    if i < 160:
                        return 160
                    return i
            return -1

        # create thresholded matrices
        hist, _ = np.histogram(lab_img[:, :, 2], bins=256, range=(0, 255))
        threshold_value = calculate_threshold_from_hist(hist)

        threshold_1 = np.logical_and((lab_img[:, :, 0] > 0), (lab_img[:, :, 0] <= 205))
        threshold_2 = np.logical_and((lab_img[:, :, 1] > 0), (lab_img[:, :, 1] <= 255))
        threshold_3 = np.logical_and(
            (lab_img[:, :, 2] > 0), (lab_img[:, :, 2] <= threshold_value)
        )

        # combine to one matrix
        th123 = np.logical_and(
            np.logical_and(threshold_1, threshold_2), threshold_3
        ).astype(np.uint8)

        # perform a simple morphological operation to smooth the binary mask
        th123 = self.smooth_binary_mask(th123)

        # create new image using threshold matrices
        new_img = self.draw_binary_mask(new_img, th123)

        cv2.imwrite(
            os.path.join(self.STARCH_AREA, os.path.basename(file_name)),
            new_img,
        )

        ground_truth = np.count_nonzero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)
        starch = np.count_nonzero(th123)

        return starch / ground_truth

    def draw_binary_mask(
        self, img: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """
        Overlays a binary mask on an image.

        Args:
            - img: The input image where the mask will be applied.
            - mask: The binary mask to be overlied on the image ().
        """
        result = img.copy()
        color = (0, 0, 0)
        alpha = 0.6
        for c in range(3):
            result[:, :, c] = np.where(
                mask == 0,
                result[:, :, c] * (1 - alpha) + alpha * color[c],
                result[:, :, c],
            )
        return result

    wdef smooth_binary_mask(self, bin_mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Smooth binary mask with basic morphological operations.
        By performing morphology, the binary mask will be smoothened to avoid discontinuity.
        """
        bin_mask = bin_mask

        # create a circular structuring element of size 10
        ksize = (10, 10)
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=ksize)

        # using to structuring element to perform one close and one open
        # operation on the binary mask
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

    def calculate_starch_multiprocessing(self, args: str) -> Any:
        file_name = args
        results = self.calculate_starch(os.path.join(self.FOLDER_NAME, file_name))
        return results

    def GrannyStarchArea(self) -> None:
        self.create_directories(self.RESULT_DIR, self.STARCH_AREA)
        image_list = os.listdir(self.FOLDER_NAME)
        cpu_count = int(os.cpu_count() * 0.8) or 1
        image_list = sorted(image_list)
        with Pool(cpu_count) as pool:
            results = pool.map(self.calculate_starch_multiprocessing, image_list)

        with open(f"{self.RESULT_DIR}{os.sep}starch_area.csv", "w") as w:
            for i, file_name in enumerate(image_list):
                w.writelines(f"{self.FOLDER_NAME}/{file_name}\t\t{results[i]}")
                w.writelines("\n")
            print('\t- Done. Check "results/" for output. - \n')
