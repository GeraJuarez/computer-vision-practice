import sys
import numpy as np
from cv2 import cv2
from cv_helpers import start_cv_video, plt_show_img
from binarization import adaptative_biarization


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

def get_contours(rgb_img, color=(255,0,255), thickness=4):
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    thresh = adaptative_biarization(gray_img)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_img = rgb_img.copy()
    index = -1 # All contours
    cv2.drawContours(new_img, contours, index, color, thickness)

    return new_img

def calculate_area_perimeter(rgb_img, color=(255,0,255), thickness=4):
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    thresh = adaptative_biarization(gray_img)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = -1
    h, w, _ = rgb_img.shape

    objects = np.zeroes([h, w, 1], 'uint8')
    for c in contours:
        cv2.drawContours(objects, [c], index, color, -1)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, closed=True)

        Moment = cv2.moments(c)
        cx = int( Moment['m10'] / Moment['m00'] )
        cy = int ( Moment['m01'] / Moment['m00'] )
        cv2.circle(objects, (cx, cy), 4, (0,0,255), -1)
        print(f'Area: {area}, Perimeter:{perimeter}')

    return objects

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
