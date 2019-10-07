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
        dilate_iter = int(params[1]) if len(params) > 1 else 0
        erode_iter = int(params[2]) if len(params) > 2 else 0

        kernel = np.ones((kernel_size,kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = cv2.erode(mask, kernel, iterations=erode_iter) 

    return mask

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RGB filter')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', help='The image to apply the filter')
    parser.add_argument('-ks', '--kernel_size', type=int, default=5, action='store', dest='kernel_size', help='The size of the kernel when applying dilatation or erotion')
    parser.add_argument('-d', '--dilate', type=int, default=0, action='store', dest='dilate_iter', help='Number of times to apply the Dilate operation')
    parser.add_argument('-e', '--erode', type=int, default=0, action='store', dest='erode_iter', help='Number of times to apply the Erode operation')
    args = parser.parse_args()

    if args.src_img:
        try:
            img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = rgb_filter(img, args.kernel_size, args.dilate_iter, args.erode_iter)
            plt_show_img(rgb_img)

        except Exception as error:
            print(error)

    else:
        start_cv_video(0, rgb_filter, args.kernel_size, args.dilate_iter, args.erode_iter)
