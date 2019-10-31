import sys
from cv_helpers import start_cv_video, plt_show_img
from cv2 import cv2

def apply_contrast_brightness(img, *params):
    alpha = params[0]
    beta = params[1]
    #alpha = 1.0 # Simple contrast control
    #beta = 0    # Simple brightness control

    new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #new_image = (img * alpha) + beta
    #new_image = cv2.addWeighted(img, alpha, img, 0, beta)
    return new_image

if __name__ == '__main__':
    try:
        alpha = float(input('* Enter the contrast value [1.0 to 3.0]: '))
        beta = int(input('* Enter the brightness value [-255 to 255]: '))
        
        if len(sys.argv) < 2:
            start_cv_video(0, apply_contrast_brightness, alpha, beta)

        else:
            img_path = sys.argv[1]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_img = apply_contrast_brightness(img, alpha, beta)
            plt_show_img(new_img)


    except Exception as error:
        print(error)
