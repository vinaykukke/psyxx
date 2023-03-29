
# Importing the OpenCV library
import cv2
# import Numpy
import numpy as np


def run_histogram_equalization(image_path):
    rgb_img = cv2.imread(image_path)
    # convert from RGB color-space to HSV
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    # print(hsv_img[: , : , 2])

    # equalize the histogram of the V channel
    hsv_img[:, :, 2] = cv2.equalizeHist(hsv_img[:, :, 2])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return equalized_img


def remove_background(image):
    # RGB Image
    rgb_img = run_histogram_equalization(image)
    # Background Image
    background = cv2.imread("space.jpeg")
    # First Convert to Grayscale
    grey_scale_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    _, baseline = cv2.threshold(grey_scale_img, 180, 255, cv2.THRESH_BINARY)
    _, foreground = cv2.threshold(baseline, 196, 255, cv2.THRESH_BINARY_INV)

    # Update foreground with bitwise_and to extract real foreground
    fo = cv2.bitwise_and(
        background,
        background,
        mask=cv2.bitwise_not(foreground)
    )
    f = cv2.bitwise_and(rgb_img, rgb_img, mask=foreground)
    ff = cv2.bitwise_or(fo, f, mask=None)

    # Combine the background and foreground to obtain our final image
    # res = np.hstack((canvas, mask))
    cv2.imshow('Weighted Image', ff)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def get_contour(image):
    # RGB Image
    rgb_img = run_histogram_equalization(image)
    # First Convert to Grayscale
    grey_scale_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    # blured = cv2.medianBlur(grey_scale_img, 3)
    # generate baseline im age without background
    _, baseline = cv2.threshold(grey_scale_img, 180, 255, cv2.THRESH_BINARY)
    # Create Canvas
    # canvas = np.zeros(rgb_img.shape)
    # canvas.fill(255)

    # # Creat Mask
    mask = np.zeros(rgb_img.shape)
    mask.fill(255)

    # Contours
    # contours, _ = cv2.findContours(
    #     baseline.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Significant Contours
    contours_mask, _ = cv2.findContours( 
        baseline.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Contour Traversal
    # for contour in range(len(contours)):
    #     cv2.drawContours(canvas, contours, contour, (0, 0, 0), 2)

    # Significant Contour Traversal
    for contour in range(len(contours_mask)):
        if (len(contours_mask[contour]) > 1500):
            cv2.fillConvexPoly(
                mask, contours_mask[contour], (0, 0, 0))

    # cv2.drawContours(canvas, contours, -1, (0, 0, 0), 2)
    # cv2.drawContours(mask, contours_mask, -1, (0, 0, 0), 2)

    # Find Canny edges
    # edges = cv2.Canny(baseline, 30, 200)
    # res = np.hstack((grey_scale_img, mask))
    cv2.imshow('Weighted Image', mask)

    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


remove_background("image.png")
