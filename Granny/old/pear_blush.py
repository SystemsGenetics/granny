"""
Created on Tue Mar 21 16:51:28 2023

@author: rene.mogollon
"""

import glob
import os

import cv2 as cv
import numpy as np
import pandas as pd


def nothig(x):
    return


def cal_blush(img_1="DSC_0519_9.png", img_2="DSC_0510_8.png", img_3="DSC_0510_10.png"):
    cv.namedWindow("Blush Calibration")
    cv.createTrackbar("a*", "Blush Calibration", 0, 255, nothig)
    cv.setTrackbarMax("a*", "Blush Calibration", 255)
    cv.setTrackbarMin("a*", "Blush Calibration", 0)
    flag_acoor = -1000

    while 1:
        color_blush = cv.getTrackbarPos("a*", "Blush Calibration")
        if flag_acoor != color_blush:
            # load images
            bgr_img_1 = cv.imread(img_1)
            bgr_img_2 = cv.imread(img_2)
            bgr_img_3 = cv.imread(img_3)

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


def blush_percentage():
    if os.path.exists(os.getcwd() + "//RGB BlushPercentage") == False:
        os.mkdir(os.getcwd() + "//RGB BlushPercentage")
    files = pd.DataFrame({"file": glob.glob("*.png")})
    blush_file = pd.DataFrame({"file": [], "Fruitpx": [], "Blushpx": [], "Blushpct": []})
    for file in files.file:
        print(file)
        bgr_image = cv.imread(file)
        lab_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2Lab)
        fruit_px = lab_image[:, :, 2] > 140
        fruit_px.sum()
        blush_px = lab_image[:, :, 1] > 148
        bgr_image[:, :, 0][blush_px] = 150
        bgr_image[:, :, 1][blush_px] = 55
        bgr_image[:, :, 2][blush_px] = 50
        blush_px.sum()
        blush_pct = 100 * blush_px.sum() / fruit_px.sum()
        cv.putText(
            bgr_image,
            "Blush:" + str(blush_pct.round(1)) + "%",
            (20, 50),
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
        cv.imwritxe(os.getcwd() + "//RGB BlushPercentage//" + "BLP_" + file, bgr_image)
    blush_file.to_csv(os.getcwd() + "//RGB BlushPercentage//Blush percentage.csv", index=False)


#### MAIN ####
## SET WORKING DIRECTORY
os.chdir("E:\\rene.mogollon\\Downloads\\segmented_data-20230322T000248Z-001\\segmented_data\\")
## BLUSH CALIBRATION PROCESS
cal_blush()
## RUN BLUSH CALCULATION OVER THE WORKING DIRECTORY FILES
blush_percentage()
