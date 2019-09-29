import cv_helpers as helpers
from cv2 import cv2

def apply_contrast(frame, *params):
    alpha = params[0]
    beta = params[1]
    #alpha = 1.0 # Simple contrast control
    #beta = 0    # Simple brightness control

    #new_image = cv2.convertScaleAbs(frame, alpha=params[0], beta=params[1])
    #new_image = (frame * params[0]) + params[1]
    new_image = cv2.addWeighted(frame, alpha, frame, 0, beta)
    return new_image

if __name__ == '__main__':
    try:
        alpha = float(input('* Enter the contrast value [1.0 to 3.0]: '))
        beta = int(input('* Enter the brightness value [-255 to 255]: ')) 

        helpers.start_video(0, apply_contrast, alpha, beta)

    except Exception as error:
        print(error)
