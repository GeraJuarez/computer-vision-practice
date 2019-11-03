import numpy as np
from cv2 import cv2
from cv_helpers import cv2_show_img, plt_show_img, start_cv_video, RGB_RED, RGB_BLUE, RGB_GREEN
from bright_contrast import apply_contrast_brightness
from binarization import otsu_binarization, grayscale_binarization
from edge_detection import apply_sobel, apply_canny, apply_prewitt

def filter(img, *params):
    kernel = np.ones((3,3),np.uint8)
    result = img.copy()

    img = apply_sobel(img)
    img = grayscale_binarization(img, threshold=10, bin_val=255)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, 2)

    result = draw_contours(img)
    result = draw_circles(img)


    return result
    '''
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1) 
    img = apply_contrast_brightness(img, 1, 20)
    img = otsu_binarization(img)
    img = cv2.dilate(img, kernel, iterations=3)
    '''

def draw_contours(img):
    # TODO: Maybe use average or std dev of areas to filter the contours

    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_cnt = []
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        perimeter_points = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        area = cv2.contourArea(cnt)
        if (10 < len(perimeter_points) < 20) and (1000 < area < 90000):
            filtered_cnt.append(cnt)
            cv2.drawContours(output, perimeter_points, -1, RGB_BLUE, 10)

    cv2.drawContours(output, filtered_cnt, -1, RGB_RED, 2)
    return output

def draw_circles(img):
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, minDist=100)#minDist=200, param1=30, param2=45, minRadius=0, maxRadius=0)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, RGB_GREEN, 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), RGB_GREEN, -1)
    
    return output

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
