import os
import tkinter
from multiprocessing import Pool
from typing import cast

import cv2
from Granny import GRANNY_Base as granny


class GrannyPearBlush(granny.GrannyBase):
    def __init__(self, action: str, fname: str):
        super().__init__(action, fname)
        self.blush_threshold = 0

    def trackbar_change(self, val: int):
        self.blush_threshold = val

    def calculate_blush_region(self, file_name: str) -> float:
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        print(f"\t- Rating {file_name}. -")
        new_img = img.copy()
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        fruit_px = lab_img[:, :, 2] > 140
        blush_px = lab_img[:, :, 1] > self.blush_threshold
        new_img[:, :, 0][blush_px] = 150
        new_img[:, :, 1][blush_px] = 55
        new_img[:, :, 2][blush_px] = 50
        blush_pct = 100 * blush_px.sum() / fruit_px.sum()
        cv2.putText(
            new_img,
            "Blush: " + str(blush_pct.round(1)) + "%",
            (20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=3,
        )
        cv2.imwrite(
            os.path.join(self.BLUSHED_IMAGES, os.path.basename(file_name)),
            new_img,
        )

        return blush_px.sum() / fruit_px.sum()

    def calibrate_blush_region(self) -> int:
        cv2.namedWindow("Calibration")
        cv2.createTrackbar("a*", "Calibration", 0, 255, self.trackbar_change)
        cv2.setTrackbarMax("a*", "Calibration", 255)
        cv2.setTrackbarMin("a*", "Calibration", 0)
        return 0

    def calculate_blush_region_multiprocessing(self, args: str) -> float:
        file_name = args
        results = self.calculate_blush_region(os.path.join(self.FOLDER_NAME, file_name))
        return results

    def GrannyPearBlush(self):
        if os.environ.get("DISPLAY") is not None:
            self.blush_threshold = self.calibrate_blush_region()
        else:
            self.blush_threshold = 148

        self.create_directories(self.RESULT_DIR)
        image_list = os.listdir(self.FOLDER_NAME)
        cpu_count = int(os.cpu_count() * 0.8) or 1
        image_list = sorted(image_list)
        with Pool(cpu_count) as pool:
            results = pool.map(self.calculate_blush_region_multiprocessing, image_list)

        with open(f"{self.RESULT_DIR}{os.sep}pear_blush.csv", "w") as w:
            for i, file_name in enumerate(image_list):
                w.writelines(f"{self.FOLDER_NAME}/{file_name}\t\t{results[i]}")
                w.writelines("\n")
            print(f'\t- Done. Check "results/" for output. - \n')
