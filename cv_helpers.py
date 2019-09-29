import sys
import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def plt_hist(data, title='', bins=256):
    plt.hist(data, bins)
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def plt_show_img(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def cv2_show_img(img):
    ANY_KEY = 0
    cv2.imshow('', img)
    cv2.waitKey(ANY_KEY)

def show_compared_imgs(img1, img2, title=''):
    img_comparison = np.concatenate((img1, img2), axis=1)
    plt_show_img(img_comparison, title)

def start_video(camera = 0, img_modifier = None, *img_mod_params):
    def get_frame(cap):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    cap = cv2.VideoCapture(camera)
    frame_view = plt.imshow(get_frame(cap), cmap='gray')

    def update(i):
        frame = get_frame(cap)
        if img_modifier is not None:
            frame = img_modifier(frame, *img_mod_params)
        frame_view.set_data(frame)

    _ = FuncAnimation(plt.gcf(), update, interval=200)
    plt.show()
