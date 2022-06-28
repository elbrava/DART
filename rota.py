import cv2
import cvzone
import numpy as np

img = cv2.imread("images.png")
h, w, c = img.shape
center = h // 2, w // 2
im = np.zeros(img.shape, np.uint8)
imz = cv2.bitwise_not(im)
img = imz - img
matrix = cv2.getRotationMatrix2D(center, 64 - 90, 0.7)

img = cv2.warpAffine(img, matrix, (w, h))

cv2.imshow("yeah", img)
cv2.imshow("yea", im)
cv2.imshow("ye", imz)
cv2.imwrite("2.jpg", img)
cv2.waitKey(10000)
