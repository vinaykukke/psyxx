
# Importing the OpenCV library
import cv2
# import Numpy
import numpy as np

def run_histogram_equalization(image_path):
    rgb_img = cv2.imread(image_path)

    # convert from RGB color-space to YCrCb
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # print(hsv_img[: , : , 2])

    # equalize the histogram of the Y channel
    hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    res = np.hstack((rgb_img, equalized_img))
    cv2.imshow('equalized_img', res)
    cv2.waitKey(0)


run_histogram_equalization("image.png")
