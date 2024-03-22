"""
Created on Tue Mar 21 16:51:28 2023

@author: rene.mogollon
"""

import glob
import os
import tkinter as tk
from tkinter import filedialog
from typing import List

import cv2 as cv
import numpy as np
import pandas as pd


def nothig(x):
    return


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()

    file_paths: List[str] = []
    while True:
        if len(file_paths) == 3:
            break
        file_path = filedialog.askopenfilename(initialdir="~", title="Select 3 Files", filetypes=(("All Files", "*.png"),))
        if file_path:
            file_paths.append(file_path)
            print("Selected file:", file_path)
        else:
            if file_paths:
                print("No more files to select.")
                break
            else:
                print("No files selected. Please select at least one file.")

    return file_paths


def cal_blush(file_paths: List[str]):
    cv.namedWindow("Blush Calibration")
    cv.createTrackbar("a*", "Blush Calibration", 0, 255, nothig)
    cv.setTrackbarMax("a*", "Blush Calibration", 255)
    cv.setTrackbarMin("a*", "Blush Calibration", 0)
    flag_acoor = -1000

    while 1:
        color_blush = cv.getTrackbarPos("a*", "Blush Calibration")
        if flag_acoor != color_blush:
            # load images
            bgr_img_1 = cv.imread(file_paths[0])
            bgr_img_2 = cv.imread(file_paths[1])
            bgr_img_3 = cv.imread(file_paths[2])

            # resize images
            rsz_bgr_img_1 = cv.resize(bgr_img_1, (500, 300))
            rsz_bgr_img_2 = cv.resize(bgr_img_2, (500, 300))
            rsz_bgr_img_3 = cv.resize(bgr_img_3, (500, 300))
            lab_img_1 = cv.cvtColor(rsz_bgr_img_1, cv.COLOR_BGR2Lab)
            lab_img_2 = cv.cvtColor(rsz_bgr_img_2, cv.COLOR_BGR2Lab)
            lab_img_3 = cv.cvtColor(rsz_bgr_img_3, cv.COLOR_BGR2Lab)
            img_arry = np.hstack((rsz_bgr_img_1, rsz_bgr_img_2, rsz_bgr_img_3))
            # create mask
            mask_img_1 = np.zeros((300, 500, 3), np.uint8)
            mask_img_2 = np.zeros((300, 500, 3), np.uint8)
            mask_img_3 = np.zeros((300, 500, 3), np.uint8)
            # select pixels of interes based on a*
            mask_img_1[lab_img_1[:, :, 1] > color_blush] = 255
            mask_img_2[lab_img_2[:, :, 1] > color_blush] = 255
            mask_img_3[lab_img_3[:, :, 1] > color_blush] = 255
            mask_array = np.hstack((mask_img_1, mask_img_2, mask_img_3))
            # show window with img + mask
            out_img = cv.addWeighted(img_arry, 0.9, mask_array, 0.7, 1)
            cv.imshow("Blush Calibration", out_img)
            flag_acoor = color_blush
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    return color_blush


def blush_percentage(blush_threshold):
    if not os.path.exists(f"{os.getcwd()}{os.sep}BlushResults"):
        os.mkdir(f"{os.getcwd()}{os.sep}BlushResults")
    files = pd.DataFrame({"file": glob.glob("*.png")})
    blush_file = pd.DataFrame({"file": [], "Fruitpx": [], "Blushpx": [], "Blushpct": []})
    for file in files.file:
        bgr_image = cv.imread(file)
        lab_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2LAB)
        fruit_px = lab_image[:, :, 2] > 140
        fruit_px.sum()
        blush_px = lab_image[:, :, 1] > blush_threshold
        bgr_image[:, :, 0][blush_px] = 150
        bgr_image[:, :, 1][blush_px] = 55
        bgr_image[:, :, 2][blush_px] = 50
        blush_px.sum()
        blush_pct = 100 * blush_px.sum() / fruit_px.sum()
        image_height, _, _ = bgr_image.shape
        cv.putText(
            bgr_image,
            "Blush:" + str(blush_pct.round(2)) + "%",
            (20, image_height - 20),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=3,
        )
        blush_file = pd.concat(
            [
                blush_file,
                pd.DataFrame(
                    {
                        "file": [file],
                        "Fruitpx": [fruit_px.sum()],
                        "Blushpx": [blush_px.sum()],
                        "Blushpct": [blush_pct],
                    }
                ),
            ]
        )
        cv.imwrite(f"{os.getcwd()}{os.sep}BlushResults{os.sep}BLP_{file}", bgr_image)
    blush_file.to_csv(f"{os.getcwd()}{os.sep}BlushResults{os.sep}Blush_percentage.csv", index=False)

file_paths = open_file_dialog()
blush_threshold = cal_blush(file_paths)
blush_percentage(blush_threshold)
