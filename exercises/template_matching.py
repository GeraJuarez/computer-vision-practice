import numpy as np
import cv2

template = cv2.imread('img_src/template.jpg', cv2.IMREAD_GRAYSCALE)
frame = cv2.imread('img_src/players.jpg', cv2.IMREAD_GRAYSCALE)
result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val,max_loc)
cv2.circle(result, max_loc, 15, 255, 2)

cv2.imshow('Frame', frame)
cv2.imshow('Template', template)
cv2.imshow('Matching', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
