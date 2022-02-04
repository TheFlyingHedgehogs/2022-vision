import numpy as np
import cv2
import glob
import pickle as pkl
from multiprocessing import pool

size = (6, 8)
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((size[0]*size[1], 3), np.float64)
# objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

objp = np.zeros((1, size[0]*size[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)

# for row in objp:
#     row[0] *= 21.6 / 1000
#     row[1] *= 21.6 / 1000

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob("calib/elp-1/*.jpg")
# imgobjpoint_queue = multiprocessing.queues.Queue()


def calib(fname: str):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, size, None)
    # If found, add object points, image points (after refining them)
    if ret:
        # objpoint_queue.put(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # imgpoint_queue.put(corners)
        # imgobjpoint_queue.put()
        # # Draw and display the corners
        cv2.drawChessboardCorners(img, size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1)
        return corners2, objp
    return []


with pool.Pool(10) as p:
    lst = list(p.map(calib, images))

    for index, item in enumerate(lst):
        if len(item) == 0:
            print(f"Failed to calibrate on {images[index]}")
            continue
        imgpts, objpts = item
        imgpoints.append(imgpts)
        objpoints.append(objpts)

    # print(np.expand_dims(np.asarray(objpoints), -2))
    # print(np.array(objpoints))

    mtx = np.zeros((3, 3))
    dist = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(imgpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(len(imgpoints))]

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    ret, _, _, _, _ = \
        cv2.fisheye.calibrate(np.asarray(objpoints), imgpoints, (1920, 1080),
                              mtx,
                              dist,
                              rvecs,
                              tvecs,
                              calibration_flags,
                              (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
    # print(ret)

    # img = cv2.imread(images[1])

    # print(rvecs)
    # print()
    # print()
    # print()
    # print(tvecs)

    # cv2.aruco.drawAxis(img, mtx, dist, rvecs[1], tvecs[1], 1)

    # cv2.imshow("axis", img)
    # cv2.imwrite("axis.png", img)
    # cv2.waitKey(10000)

    print(f"Calibrated with {len(objpoints)} images")

    img = cv2.imread(images[0])
    # undistorted = cv2.fisheye.undistortImage(img, mtx, dist)

    h, w = img.shape[:2]
    K = mtx
    D = dist
    DIM = np.array([1920, 1080])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("undistorted", undistorted)
    cv2.waitKey(5000)
    cv2.imwrite("ifjeowfoew.png", undistorted)

    cv2.destroyAllWindows()

    with open("calib/elp-1/calib.pkl", "wb") as f:
        pkl.dump((mtx, dist, rvecs, tvecs), f)
