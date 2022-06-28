import cv2
import numpy as np

# Read image
img = cv2.imread('dart.jpg')
hh, ww = img.shape[:2]

# threshold on white
# Define lower and uppper limits
lower = np.array([200, 200, 200])
upper = np.array([255, 255, 255])

# Create mask to only select black
thresh = cv2.inRange(img, lower, upper)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morp image
mask = 255 - morph

# apply mask to image
result = cv2.bitwise_and(img, img, mask=thresh)

result = img - result
alpha = np.full((hh, ww), 0, dtype=np.uint8)
result = np.dstack((result, alpha))

# save results

cv2.imshow('result', result)
cv2.imwrite("result.png", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
