import sys
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

def plt_hist(data, title='', bins=256):
    plt.hist(data, bins)
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def plt_show_img(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def show_compared_imgs(img1, img2, title=''):
    img_comparison = np.concatenate((img1, img2), axis=1)
    plt_show_img(img_comparison, title)

def otsu_binarization(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Missing arguments')

    try:
        img_path = sys.argv[1]
        img = cv2.imread(img_path, 0)
        ret, imgf = otsu_binarization(img)

        show_compared_imgs(img, imgf, 'Orginal vs Otsu Binarization')
        if (len(sys.argv) == 3):
            plt_hist(img.ravel(), 'Color Histogram')

    except Exception as error:
        print(error)
