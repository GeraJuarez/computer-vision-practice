import sys
import cv_helpers as helpers
from cv2 import cv2

def otsu_binarization(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception('Missing file path argument')

    try:
        img_path = sys.argv[1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ret, imgf = otsu_binarization(img)

        helpers.show_compared_imgs(img, imgf, 'Orginal vs Otsu Binarization')
        if (len(sys.argv) == 3):
            helpers.plt_hist(img.ravel(), 'Color Histogram')

    except Exception as error:
        print(error)
