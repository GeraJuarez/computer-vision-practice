import sys
import numpy as np
from cv_helpers import plt_hist, start_cv_video, plt_show_img
from cv2 import cv2

def otsu_grayscale_binarization(img): 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _ , img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def otsu_binarization(img):
    _, binarized_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized_img

def mask_binarization(img):
    _, binarized_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = np.where(binarized_img==255, 1, binarized_img)
    return mask

if __name__ == '__main__':
    if len(sys.argv) < 2:
        start_cv_video(img_filter=otsu_grayscale_binarization)
    
    else:
        try:
            img_path = sys.argv[1]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            binarized_img = otsu_grayscale_binarization(img)
            plt_show_img(binarized_img)

            if (sys.argv[2] == '-h'):
                plt_hist(img.ravel(), 'Color Histogram')

        except Exception as error:
            print(error)
