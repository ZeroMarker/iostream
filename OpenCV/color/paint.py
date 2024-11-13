import numpy as np
import cv2

img = np.zeros((600, 600, 3), np.uint8)
# Draw a line, rectangle and put a text on the image.
cv2.line(img, (0, 0), (300, 300), (255, 0, 0), 1)
cv2.rectangle(img, (0, 0), (200, 200), (0, 0, 255), cv2.FILLED)
cv2.putText(img, 'Hello', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

cv2.imshow('img', img)
cv2.waitKey(0)
