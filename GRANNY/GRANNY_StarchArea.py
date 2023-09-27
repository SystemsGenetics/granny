import os
from multiprocessing import Pool
from typing import Any, List, Tuple, cast

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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(f"\t- Rating {file_name}. -")

        thresh = np.logical_and(gray_img > 5, gray_img < 128).astype(np.uint8) * 255

        img = gray_img * thresh

        cv2.imwrite(
            os.path.join(self.STARCH_AREA, os.path.basename(file_name)),
            img,
        )
        return np.count_nonzero(thresh) / np.count_nonzero(gray_img)

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
            print(f'\t- Done. Check "results/" for output. - \n')
