import cv2
import numpy as np

from stages import mtx, dist

img = cv2.imread("images/6m-0m-0d.png")
# img = cv2.imread("testimg.png")

# undistorted = cv2.undistort(img, mtx, dist)

h, w = img.shape[:2]
K = mtx
D = dist
DIM = np.array([1920, 1080])
map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imshow("undistorted", undistorted)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
cv2.waitKey(10000000)
