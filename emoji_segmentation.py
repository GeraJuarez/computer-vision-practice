import numpy as np
from cv2 import cv2
from cv_helpers import cv2_show_img, plt_show_img, start_cv_video
from bright_contrast import apply_contrast_brightness
from binarization import otsu_binarization, grayscale_binarization
from edge_detection import apply_sobel, apply_canny, apply_prewitt
# increase contrast
# gauusian and grayscale
# closing

def filter(img, *params):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    saturated = increase_sat(img)
    saturated_g = cv2.cvtColor(saturated, cv2.COLOR_RGB2GRAY)

    kernel = np.ones((3,3),np.uint8)

    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 2)
    return img# * saturated_g

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1) 
    img = apply_contrast_brightness(img, 1, 20)
    img = otsu_binarization(img)
    img = cv2.dilate(img, kernel, iterations=3)

def increase_sat(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype('float32')
    h, s, v = cv2.split(img)
    s = s + 200
    s = np.clip(s,0,255)
    imghsv = cv2.merge((h,s,v))
    imghsv = cv2.cvtColor(imghsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return imghsv

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Emoji segmentation')
    parser.add_argument('-i', '--image', type=str, action='store', dest='src_img', help='The image to apply the filter')
    parser.add_argument('-ks', '--kernel_size', type=int, default=5, action='store', dest='kernel_size', help='The size of the kernel when applying dilatation or erotion')
    parser.add_argument('-d', '--dilate', type=int, default=0, action='store', dest='dilate_iter', help='Number of times to apply the Dilate operation')
    parser.add_argument('-e', '--erode', type=int, default=0, action='store', dest='erode_iter', help='Number of times to apply the Erode operation')
    args = parser.parse_args()

    if args.src_img:
        try:
            img = cv2.imread(args.src_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_img = filter(img)
            plt_show_img(rgb_img)

        except Exception as error:
            print(error)

    else:
        start_cv_video(0, filter)
