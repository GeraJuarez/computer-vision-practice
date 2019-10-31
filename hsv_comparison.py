import numpy as np
import cv2

from cv_helpers import cv2_show_img

def compare_hsv_channels(img):
    h, s, v = cv2.split(img)
    hsv_split = np.concatenate((h,s,v), axis=1)
    cv2_show_img(hsv_split)
