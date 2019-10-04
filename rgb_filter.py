import numpy as np
from cv2 import cv2
from cv_helpers import cv2_show_img, plt_show_img, start_cv_video
from binarization import otsu_binarization

def rgb_filter(img, *params):
    red, green, blue = cv2.split(img)
    binarized_r = otsu_binarization(red)
    binarized_g = otsu_binarization(green)
    binarized_b = otsu_binarization(blue)

    #binarized_r = np.where(binarized_r==255, 0, binarized_r)
    #binarized_g = np.where(binarized_g==255, 0, binarized_g)
    #binarized_b = np.where(binarized_b==255, 0, binarized_b)
    mask = cv2.merge( (binarized_r, binarized_g, binarized_b) )

    if len(params) > 0:
        kernel_size = int(params[0]) if len(params) > 0 else 5
        dilate_iter = int(params[1]) if len(params) > 1 else 1
        erode_iter = int(params[2]) if len(params) > 2 else 1

        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = cv2.erode(mask, kernel, iterations=erode_iter) 

    return mask

if __name__ == '__main__':
    import sys

    argv_size = len(sys.argv)
    if sys.argv[1] == '-c':
        if argv_size > 2:
            params = sys.argv[2:argv_size]
            start_cv_video(0, rgb_filter, *params)
        else:
            start_cv_video(0, rgb_filter)

    else:
        try:
            img_path = sys.argv[1]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if argv_size > 2:
                params = sys.argv[2:argv_size]
                rgb_img = rgb_filter(img, *params)    
            else: 
                rgb_img = rgb_filter(img)
            plt_show_img(rgb_img)

        except Exception as error:
            print(error)
