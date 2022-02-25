import numpy as np
import cv2 as cv
import glob
import pickle as pkl
from multiprocessing import pool

shape = 7, 7

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((shape[0] * shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:shape[0], 0:shape[1]].T.reshape(-1, 2)
# objp = np.zeros((6*8, 3), np.float32)
# objp[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
for row in objp:
    # row[0] *= 21.6 / 1000
    # row[1] *= 21.6 / 1000
    row[0] *= 0.25
    row[1] *= 0.25

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
filenames = []  # 2d points in image plane.
images = glob.glob("calib/virtual-camera-2/*.png")


# images = glob.glob("calib/picam-1/*.png")
# imgobjpoint_queue = multiprocessing.queues.Queue()
# inp = cv.VideoCapture("calib/picam-1/video-slow.mkv")
# images = []
# ret = True
# while ret:
#     ret, img = inp.read()
#     if ret:
#         images.append(img)


def calib(fname: str):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, shape, None)
    # ret, corners = cv.findChessboardCorners(gray, (6, 8), None)
    # If found, add object points, image points (after refining them)
    if ret:
        # objpoint_queue.put(objp)
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        # imgpoint_queue.put(corners)
        # imgobjpoint_queue.put()
        print(".", end="")
        # return corners2, objp
        # # Draw and display the corners
        # cv.drawChessboardCorners(img, (7, 7), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(1000)
        return corners2, objp, fname
    print("-", end="")
    return []


with pool.Pool(4) as p:
    lst = list(p.map(calib, images))

    for item in lst:
        if len(item) == 0:
            continue
        imgpts, objpts, fname = item
        imgpoints.append(imgpts)
        objpoints.append(objpts)
        filenames.append(fname)

    # mtx = np.zeros((3, 3))
    # dist = np.zeros((4, 1))
    # rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(imgpoints))]
    # tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(imgpoints))]

    ret, mtx, dist, rvecs, tvecs = \
        cv.calibrateCamera(objpoints, imgpoints, (1920, 1080),
                           None,
                           None)

    print(tvecs[filenames.index("calib/virtual-camera-2/0041.png")])
    # cv.destroyAllWindows()

    with open("calib/virtual-camera-2/calib.pkl", "wb") as f:
        # with open("calib/picam-1/calib.pkl", "wb") as f:
        pkl.dump((mtx, dist, rvecs, tvecs), f)
