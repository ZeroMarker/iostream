import cv2
import numpy as np
import random
# img = cv2.imread('color.jpg')
#
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow('video', frame)
#     else:
#         break
#     if cv2.waitKey(10) == ord('q'):
#         break

# img = cv2.imread('color.jpg')
# print(img.shape)

# img = np.empty((300, 300, 3), np.uint8)
img = cv2.imread('color.jpg')
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
# for row in range(300):
#     for column in range(img.shape[1]):
#         img[row][column] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
# newImg = img[:150, :200]
kernel = np.ones((3, 3), np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (3, 3), 9)
canny = cv2.Canny(img, 150, 200)
dilate = cv2.dilate(canny, kernel, iterations=1)
erode = cv2.erode(dilate, kernel, iterations=1)
cv2.imshow('img', img)
cv2.imshow('blur', blur)
cv2.imshow('canny', canny)
cv2.imshow('dilate', dilate)
cv2.imshow('erode', erode)

cv2.waitKey(0)
