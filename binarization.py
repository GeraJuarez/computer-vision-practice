import sys
import numpy as np
from cv_helpers import plt_hist, start_cv_video, plt_show_img
from cv2 import cv2

def grayscale_binarization(img, threshold=127, bin_val=255): 
    _ , img = cv2.threshold(img, threshold, bin_val, cv2.THRESH_BINARY)
    return img

def otsu_binarization(img, bin_val=255):
    _, binarized_img = cv2.threshold(img, 0, bin_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized_img

def mask_binarization(img):
    _, binarized_img = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized_img

def adaptative_biarization(img, max_val=255, neighboor_param=115):
    binarized_img = cv2.adaptiveThreshold(img, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighboor_param, 1)
    return binarized_img

def slow_binarization(img, threshold=127):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = img.shape
    binary = np.zeros([height, width, 1], 'uint8')

    for row in range(0, height):
        for col in range(0, width):
            if img[row][col] > threshold:
                binary[row][col] = 255

    return binary


if __name__ == '__main__':
    if len(sys.argv) < 2:
        start_cv_video(img_filter=otsu_binarization)
    
    else:
        try:
            img_path = sys.argv[1]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            binarized_img = otsu_binarization(img)
            plt_show_img(binarized_img)

            if (sys.argv[2] == '-h'):
                plt_hist(img.ravel(), 'Color Histogram')

        except Exception as error:
            print(error)
