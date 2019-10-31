import numpy as np
import cv2 as cv2
import random

img = cv2.imread("fuzzy.png", 1)
cv2.imshow('Original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
cv2.imshow('Binary', thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

filtered = []
for c in contours:
    if cv2.contourArea(c) < 1000: continue
    filtered.append(c)

objects = np.zeros([img.shape[0], img.shape[1], 3], 'uint8')
for c in filtered:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    cv2.drawContours(objects, [c], -1, color, -1) # first -1 to draw all objs, and last -1 to fill the obj  

cv2.imshow('COntorus', objects)

cv2.waitKey(0)
cv2.destroyAllWindows()