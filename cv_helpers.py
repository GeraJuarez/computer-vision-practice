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
