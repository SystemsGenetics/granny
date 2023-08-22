import os
from typing import List, Tuple

import cv2
import numpy as np
import skimage
from GRANNY import GRANNY_Base as granny
from numpy.typing import NDArray


class GrannyPeelColor(granny.GrannyBase):
    def __init__(self, action: str, fname: str, num_instances: int):
        num_instances = 1 if num_instances == None else num_instances
        super(GrannyPeelColor, self).__init__(action, fname, num_instances)

        """ Raw reference colors """
        # self.MEAN_VALUES_L = [
        #     65.46409053564173, 65.93893026633565, 69.13779435809776, 71.43683033604663, 70.45811857450016,
        #     73.47674074368683, 76.35187387512626, 78.16686318517812, 80.02421866187458, 79.88775577232568
        # ]
        # self.MEAN_VALUES_A = [
        #     -29.644531843745256, -30.650149465470946, -28.892427531452014, -22.53765384435397, -15.90757099266203,
        #     -19.56968650580444, -17.071513900250608, -13.695975543688576, -11.488617337829794, -10.959318652083951
        # ]
        # self.MEAN_VALUES_B = [
        #     45.947826417388654, 48.302078583371326, 58.190157230064415, 60.87525738218915, 61.80401151106037,
        #     63.52708437720722, 63.344500279173644, 64.09726801331166, 68.80767308073794, 67.3652391861267
        # ]

        self.MEAN_VALUES_A: List[float] = [
            -32.44750225,
            -31.83257147,
            -25.9446722,
            -21.09812112,
            -16.97274929,
            -18.13955358,
            -16.84993067,
            -14.61041991,
            -11.04855559,
            -11.47330226,
        ]
        self.MEAN_VALUES_B: List[float] = [
            49.06275021,
            49.6160969,
            54.91433483,
            59.27551351,
            62.98773761,
            61.93778644,
            63.09825619,
            65.11348442,
            68.31863516,
            67.93642601,
        ]
        self.SCORE: List[float] = [
            0.626914290076815,
            0.6339848456157803,
            0.7016846957558457,
            0.7574109873047857,
            0.8048450640691405,
            0.7914289917882015,
            0.8062572485320575,
            0.8320074424392314,
            0.8729622333028093,
            0.868078439684183,
        ]
        self.SCORE: List[float] = [
            0.5192001330394723,
            0.5233446426876467,
            0.5838859128997311,
            0.6529992071684837,
            0.7302285740121869,
            0.6934065834794143,
            0.7210722038415041,
            0.7652909029091124,
            0.8066780913327652,
            0.8106974639404376,
        ]

        self.LINE_POINT_1 = np.array([-86.97064, 0])
        self.LINE_POINT_2 = np.array([0, 78.2607])

        """ Scaled reference colors """
        # self.MEAN_VALUES_L = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        # self.MEAN_VALUES_A = [
        #     -37.46455193845067, -38.33945889256972, -35.856673734593976, -28.66620337913712, -20.153556829155015,
        #     -25.183705217005663, -22.73535185450301, -18.57975296251061, -15.422998269835011, -14.839191848288477
        # ]
        # self.MEAN_VALUES_B = [
        #     58.068541555881474, 60.41978876347979, 72.21634388774689, 77.42875637928557, 78.30049344632643,
        #     81.7513027496333, 84.36038598038662, 86.95338287223824, 92.37148315325989, 91.21422051166374
        # ]

        # self.MEAN_VALUES_A = [
        #     -36.64082458, -35.82390694, -29.47956688, -24.68504792, -21.51960279,
        #     -21.49440178, -19.49577289, -16.92159296, -13.70076143, -13.34873991,
        # ]
        # self.MEAN_VALUES_B = [
        #     57.4946451, 58.6671866, 67.77337014, 74.65505828, 79.19849765,
        #     79.23466925, 82.10334927, 85.79813151, 90.42106829, 90.92633324,
        # ]

        # self.LINE_POINT_1 = np.array([-76.69774, 0])
        # self.LINE_POINT_2 = np.array([0, 110.0861])
        # self.rgb = 0

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

    def get_green_yellow_values(
        self, img: NDArray[np.uint8]
    ) -> Tuple[float, float, float]:
        """
        Get the mean pixel values from the images representing the amount of
        green and yellow in the CIELAB color space. Then, normalize the values to L = 50.
        """
        # convert from RGB to Lab color space
        new_img = img
        lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float32)

        # get channel 2 histogram for min and max values
        lab_img = lab_img.astype(np.uint8)

        # create binary matrix (ones and zeros)
        bin = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)

        # set max and min values for each channel
        channel1Min = 0 * bin
        channel1Max = 255 * bin
        channel2Min = 0 * bin
        channel2Max = 128 * bin
        channel3Min = 128 * bin
        channel3Max = 255 * bin

        # create threshold matrices for each for each channel
        threshold_1 = np.greater(lab_img[:, :, 0], channel1Min) & np.less(
            lab_img[:, :, 0], channel1Max
        )
        threshold_2 = np.greater(lab_img[:, :, 1], channel2Min) & np.less(
            lab_img[:, :, 1], channel2Max
        )
        threshold_3 = np.greater(lab_img[:, :, 2], channel3Min) & np.less(
            lab_img[:, :, 2], channel3Max
        )
        th123 = threshold_1 & threshold_2 & threshold_3

        # apply the binary mask on the image
        for i in range(3):
            new_img[:, :, i] = new_img[:, :, i] * th123

        # get mean values from each channel
        mean_l = (
            np.sum(lab_img[:, :, 0])
            / np.count_nonzero(threshold_1)
            * 100
            / 255
        )
        mean_a = (
            np.sum(lab_img[:, :, 1] * threshold_2)
            / np.count_nonzero(threshold_2)
            - 128
        )
        mean_b = (
            np.sum(lab_img[:, :, 2] * threshold_3)
            / np.count_nonzero(threshold_3)
            - 128
        )

        # normalize by shifting point in the spherical coordinates
        radius = np.sqrt(mean_l**2 + mean_a**2 + mean_b**2)
        scaled_l = 50
        scaled_a = np.sign(mean_a) * np.sqrt(
            np.abs(radius**2 - scaled_l**2) / (1 + (mean_b / mean_a) ** 2)
        )
        scaled_b = np.sign(mean_b) * mean_b / mean_a * scaled_a

        return scaled_l, scaled_a, scaled_b
        return mean_l, mean_a, mean_b

    def calculate_bin_distance(
        self, color_list: List[int], method: str = "Euclidean"
    ) -> Tuple[int, NDArray[np.float16]]:
        """
        Calculate the Euclidean distance from normalized image's LAB to each
        bin color.
        Return the shortest distance and the corresponding bin.

        """
        bin_num = 0
        dist: NDArray[np.float16]
        if method == "Euclidean":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt(
                (dist_a / np.linalg.norm(dist_a)) ** 2
                + (0.2 * dist_b / np.linalg.norm(dist_b)) ** 2
            )
            bin_num = np.argmin(dist) + 1
        if method == "X-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_a / np.linalg.norm(dist_a)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Y-component":
            dist_a = color_list[0] - np.array(self.MEAN_VALUES_A)
            dist_b = color_list[1] - np.array(self.MEAN_VALUES_B)
            dist = np.sqrt((dist_b / np.linalg.norm(dist_b)) ** 2)
            bin_num = np.argmin(dist) + 1
        if method == "Score":
            dist = color_list[0] - np.array(self.SCORE)
            dist = np.abs(dist)
            bin_num = np.argmin(dist) + 1
        return bin_num, dist

    def calculate_score_distance(
        self, color_list: List[float]
    ) -> Tuple[float,]:
        """ """
        score = 0
        distance = 0
        point = 0
        color_point = np.array([color_list[1], color_list[2]])
        n = self.LINE_POINT_2 - self.LINE_POINT_1
        n /= np.linalg.norm(n)
        projection = self.LINE_POINT_1 + n * np.dot(
            color_point - self.LINE_POINT_1, n
        )
        score = np.linalg.norm(
            projection - self.LINE_POINT_1
        ) / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1)
        distance = np.linalg.norm(
            np.cross(
                self.LINE_POINT_2 - self.LINE_POINT_1,
                color_point - self.LINE_POINT_1,
            )
        ) / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1)
        point = np.sign(color_point[1] - projection[1])
        print(f"Old Score: {score}")
        score = score - point * (
            0.6
            * distance
            / np.linalg.norm(self.LINE_POINT_2 - self.LINE_POINT_1)
        )
        print(f"New Score: {score}")
        print(f"Old Coordinates: {color_list}")
        print(f"New Coordinates: {projection}")
        print(f"Distance from line: {distance}")
        print(f"Above/Under: {point}")
        if score < 0:
            score = np.float16(0)
        elif score > 1:
            score = np.float16(1.0)
        return projection, score, distance, point

    def extract_green_yellow_values(self):
        """
        Main method performing rating for peel color

        This is the main method being called by the Python argument parser from the command.py to set up CLI for
        pear's peel color rating.
        The results will be written to a .csv file, containing necessary information such as scores, file names,
        color card number, distances, and LAB mean pixels.
        The calculated scores will be written to a .csv file.

        """
        # create "results" directory to save the results
        self.create_directories(self.BIN_COLOR)

        if self.NUM_INSTANCES == 1:
            try:
                # create "results" directory to save the results
                self.create_directories(self.RESULT_DIR)

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

                # calculate distance to the least-mean-square line
                (
                    projection,
                    score,
                    orth_distance,
                    point,
                ) = self.calculate_score_distance([l, a, b])

                # calculate distance to each bin
                bin_num, distance = self.calculate_bin_distance(
                    [projection[0], projection[1]]
                )

                # save the scores to results/rating.csv
                with open(
                    self.BIN_COLOR + os.sep + "peel_colors.csv", "w"
                ) as w:
                    w.writelines(
                        f"{self.clean_name(file_name.split(os.sep)[-1])},{bin_num},{score},{str(orth_distance)},{point},{l},{a},{b}"
                    )
                    w.writelines("\n")

                print(f'\t- Done. Check "results/" for output. - \n')

            except FileNotFoundError:
                print(
                    f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -"
                )

        else:
            try:
                # list all files and folders in the folder
                folders, files = self.list_all(self.FOLDER_NAME)

                # create "results" directory to save the results
                for folder in folders:
                    self.create_directories(
                        folder.replace(self.FOLDER_NAME, self.BIN_COLOR)
                    )

                bin_nums = []
                orth_distances = []
                channels_values = []
                points = []
                ratings = []

                for file_name in files:
                    print(f"\t- Rating {file_name}. -")

                    img = skimage.io.imread(file_name)

                    # remove surrounding purple
                    img = self.remove_purple(img)
                    nopurple_img = img

                    # image smoothing
                    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0, sigmaY=0)

                    # get image values
                    l, a, b = self.get_green_yellow_values(img)

                    # calculate distance to the least-mean-square line
                    (
                        projection,
                        score,
                        orth_distance,
                        point,
                    ) = self.calculate_score_distance([l, a, b])

                    # # calculate distance to each bin
                    bin_num, distance = self.calculate_bin_distance(
                        [score], method="Score"
                    )

                    # # calculate distance to each bin
                    # bin_num, distance = self.calculate_bin_distance([projection[0], projection[1]])

                    bin_nums.append(bin_num)
                    ratings.append(score)
                    points.append(point)
                    orth_distances.append(str(orth_distance))
                    channels_values.append(
                        str(l) + "," + str(a) + "," + str(b)
                    )

                with open(
                    self.BIN_COLOR + os.sep + "peel_colors.csv", "w"
                ) as w:
                    for i in range(len(bin_nums)):
                        w.writelines(
                            f"{self.clean_name(files[i].split(os.sep)[-1])},{bin_nums[i]},{ratings[i]},{orth_distances[i]},{points[i]},{channels_values[i]}"
                        )
                        w.writelines("\n")
                print(f'\t- Done. Check "results/" for output. - \n')
            except FileNotFoundError:
                print(
                    f"\t- Folder/File Does Not Exist or Wrong NUM_INSTANCES Values. -"
                )
