import numpy as np
import cv2 as cv
import glob
import pickle as pkl
import multiprocessing
from multiprocessing import pool

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*5, 3), np.float32)
objp[:, :2] = np.mgrid[0:5, 0:5].T.reshape(-1, 2)
for row in objp:
    row[0] *= 0.25
    row[1] *= 0.25

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob("calib/virtual-camera-1/*.png")
# imgobjpoint_queue = multiprocessing.queues.Queue()


def calib(fname: str):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (5, 5), None)
    # If found, add object points, image points (after refining them)
    if ret:
        # objpoint_queue.put(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # imgpoint_queue.put(corners)
        # imgobjpoint_queue.put()
        return corners2, objp
        # # Draw and display the corners
        # cv.drawChessboardCorners(img, (5, 5), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
    return []


with pool.Pool(10) as p:
    lst = list(p.map(calib, images))

    for item in lst:
        if len(item) == 0:
            continue
        imgpts, objpts = item
        imgpoints.append(imgpts)
        objpoints.append(objpts)

    ret, mtx, dist, rvecs, tvecs = \
        cv.calibrateCamera(objpoints, imgpoints, (1920, 1080), None, None)
    cv.destroyAllWindows()

    with open("calib/virtual-camera-1/calib.pkl", "wb") as f:
        pkl.dump((mtx, dist, rvecs, tvecs), f)
