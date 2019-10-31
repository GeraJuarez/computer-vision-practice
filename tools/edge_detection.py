import sys
import numpy as np
from cv2 import cv2
from cv_helpers import start_cv_video, plt_show_img


def apply_sobel(img, *params):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    kernel_size = 3

    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    new_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #new_image = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)

    return new_image

def apply_prewitt(img, *params):
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    prewittx = cv2.filter2D(img, -1, kernel_x)
    prewitty = cv2.filter2D(img, -1, kernel_y)

    return prewittx + prewitty

def apply_canny(img, *params):
    low_threshold = 100
    high_threshold = 200
    img = cv2.Canny(img, low_threshold, high_threshold)

    return img

if __name__ == '__main__':
    try:
        edge_filter_dict = {
            'sobel': apply_sobel,
            'prewitt': apply_prewitt,
            'canny': apply_canny,
        }
        filter_func = edge_filter_dict.get(sys.argv[1])

        if len(sys.argv) < 3 and filter_func:
            start_cv_video(img_filter=filter_func)

        elif len(sys.argv) < 4 and filter_func:
            img_path = sys.argv[2]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_img = filter_func(img)
            plt_show_img(new_img)

        else:
            raise Exception('Method not implemented')
        
    except Exception as error:
        print(error)
        print('Usage: python3 edge_detection[canny | sobel | prewitt] [file_path]')
