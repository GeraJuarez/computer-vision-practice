import sys
import numpy as np

from cv_helpers import show_compared_imgs, plt_show_img
from binarization import otsu_binarization, mask_binarization
from edge_detection import apply_prewitt
from cv2 import cv2

if __name__ == '__main__':
    try: 
        img_path = sys.argv[1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        binarized_img = otsu_binarization(img)
        edges_img = apply_prewitt(img)

        normalized_mask = mask_binarization(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        binary_mask = edges_img * normalized_mask

        show_compared_imgs(edges_img, binarized_img, binary_mask, 'Edges, Binary, Mask')

    except Exception as error:
        print(error)
