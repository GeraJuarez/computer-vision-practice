import cv_helpers as helpers
import numpy as np
from cv2 import cv2


def apply_sobel(frame, *params):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    kernel_size = 3

    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(frame, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(frame, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)


    new_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #new_image = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)

    return new_image

def apply_prewitt(frame, *params):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    prewittx = cv2.filter2D(frame, -1, kernel_x)
    prewitty = cv2.filter2D(frame, -1, kernel_y)

    return prewittx + prewitty

def apply_canny(frame, *params):
    low_threshold = 100
    high_threshold = 200
    frame = cv2.Canny(frame, low_threshold, high_threshold)

    return frame

if __name__ == '__main__':
    try:
        helpers.start_video(0, apply_canny)

    except Exception as error:
        print(error)
