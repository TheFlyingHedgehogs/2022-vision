import math
import statistics
from math import cos, pi, sin
import cv2
import numpy as np
import dataclasses
import pickle as pkl
import utils

thresh_cache = None
hsv_cache = None


def find_filter_contours(img):
    global thresh_cache, hsv_cache

    # utils.timeit("hsv", True)
    hsv_cache = cv2.cvtColor(img, cv2.COLOR_RGB2HSV, hsv_cache)
    # utils.timeit("hsv")
    # utils.timeit("inRange", True)
    thresh_cache = cv2.inRange(hsv_cache, (0, 100, 100), (100, 255, 255), thresh_cache)
    # thresh_cache = cv2.inRange(hsv_cache, (10, 100, 70), (100, 255, 255), thresh_cache)
    # utils.timeit("inRange")
    # utils.timeit("contours", True)
    contours, _ = cv2.findContours(thresh_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # utils.timeit("contours")
    # utils.timeit("filter", True)
    output = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 75:
            continue
        output.append(c)
    # utils.timeit("filter")
    # im2 = img.copy()
    # cv2.drawContours(im2, contours, -1, (255, 0, 0))
    # cv2.imshow("contours", im2)
    return output


@dataclasses.dataclass
class Point:
    x: float
    y: float


theta = -pi / 4
rotation_matrix = np.array([
    [cos(theta), -sin(theta)],
    [sin(theta), cos(theta)],
])
inv_rotation_matrix = np.array([
    [cos(-theta), -sin(-theta)],
    [sin(-theta), cos(-theta)],
])


def find_corners(contours, img):
    corners = []
    for contour in contours:
        contour = np.squeeze(contour)
        rotated = np.matmul(contour, rotation_matrix)
        leftbottom_rot = rotated[0]
        righttop_rot = rotated[0]
        lefttop_rot = rotated[0]
        rightbottom_rot = rotated[0]
        for item in rotated:
            if item[0] < leftbottom_rot[0]:
                leftbottom_rot = item
            if item[0] > righttop_rot[0]:
                righttop_rot = item
            if item[1] < lefttop_rot[1]:
                lefttop_rot = item
            if item[1] > rightbottom_rot[1]:
                rightbottom_rot = item

        lb = np.matmul(leftbottom_rot, inv_rotation_matrix)
        rt = np.matmul(righttop_rot, inv_rotation_matrix)
        lt = np.matmul(lefttop_rot, inv_rotation_matrix)
        rb = np.matmul(rightbottom_rot, inv_rotation_matrix)

        # img2 = img.copy()
        # cv2.drawMarker(img2, (int(lb[0]), int(lb[1])), (255, 255, 0))
        # cv2.drawMarker(img2, (int(rb[0]), int(rb[1])), (255, 0, 255))
        # cv2.drawMarker(img2, (int(lt[0]), int(lt[1])), (255, 255, 255))
        # cv2.drawMarker(img2, (int(rt[0]), int(rt[1])), (0, 255, 0))
        # cv2.imshow("corner", img2)
        corners.append([
            Point(int(lb[0]), int(lb[1])),
            Point(int(rb[0]), int(rb[1])),
            Point(int(rt[0]), int(rt[1])),
            Point(int(lt[0]), int(lt[1])),
        ])

    # return corners

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    converted_corners = np.zeros((len(corners * 4), 2), dtype=np.float32)
    for i in range(0, len(corners)):
        converted_corners[i * 4][0] = corners[i][0].x
        converted_corners[i * 4][1] = corners[i][0].y

        converted_corners[i * 4 + 1][0] = corners[i][1].x
        converted_corners[i * 4 + 1][1] = corners[i][1].y

        converted_corners[i * 4 + 2][0] = corners[i][2].x
        converted_corners[i * 4 + 2][1] = corners[i][2].y

        converted_corners[i * 4 + 3][0] = corners[i][3].x
        converted_corners[i * 4 + 3][1] = corners[i][3].y

    # cv2.imshow("gray", gray)

    refined = cv2.cornerSubPix(gray, converted_corners, (11, 11), (-1, -1), criteria)

    output = []
    for i in range(0, int(len(refined) / 4)):
        output.append([
            Point(refined[i * 4][0], refined[i * 4][1]),
            Point(refined[i * 4 + 1][0], refined[i * 4 + 1][1]),
            Point(refined[i * 4 + 2][0], refined[i * 4 + 2][1]),
            Point(refined[i * 4 + 3][0], refined[i * 4 + 3][1]),
        ])

    # print("AAAAAA")
    # print(corners)
    # print(output)

    # img2 = cv2.resize(img, (1920 * 4, 1080 * 4), interpolation=cv2.INTER_NEAREST)
    # for quad in output:
    #     cv2.drawMarker(img2, (int(quad[0].x * 4), int(quad[0].y * 4)), (0, 0, 255))
    #     cv2.drawMarker(img2, (int(quad[1].x * 4), int(quad[1].y * 4)), (0, 255, 0))
    #     cv2.drawMarker(img2, (int(quad[2].x * 4), int(quad[2].y * 4)), (0, 255, 255))
    #     cv2.drawMarker(img2, (int(quad[3].x * 4), int(quad[3].y * 4)), (255, 0, 0))
    # for quad in corners:
    #     cv2.drawMarker(img2, (int(quad[0].x * 4), int(quad[0].y * 4)), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS)
    #     cv2.drawMarker(img2, (int(quad[1].x * 4), int(quad[1].y * 4)), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS)
    #     cv2.drawMarker(img2, (int(quad[2].x * 4), int(quad[2].y * 4)), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS)
    #     cv2.drawMarker(img2, (int(quad[3].x * 4), int(quad[3].y * 4)), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS)
    # cv2.imshow("corners", img2)

    # cv2.imshow("img", img)

    return output


w = 0.063405 * 2
# # w = 5 * 2.54 / 100
h = 2 * 2.54 / 100
# w = 1.0
# h = 1.0

real_coords = np.array([
    [0, 0, 0],  # left bottom
    [w, 0, 0],  # right bottom
    [w, h, 0],  # right top
    [0, h, 0]   # left top
], dtype="double")

for triple in real_coords:
    triple[0] -= w / 2
    triple[2] += 0.674882

# /home/pi/2898-2022-vision-py/
# mtx, dist, rvecs, tvecs = pkl.load(open("calib/picam-1/calib.pkl", "rb"))
mtx, dist, rvecs, tvecs = pkl.load(open("calib/virtual-camera-2/calib.pkl", "rb"))

# tilt_angle = math.radians(9.33)
tilt_angle = math.radians(-20)
# tilt_angle = math.radians(16.5)
# tilt_angle = 0.0

# x
tilt_matrix = np.array([
    [1, 0, 0],
    [0, cos(-tilt_angle), -sin(-tilt_angle)],
    [0, sin(-tilt_angle), cos(-tilt_angle)]
])


def compute_output_values(rvec, tvec):
    """
    Computes the distance and angle to the target.

    :param rvec: The solvepnp rotation vector, unused as of now.
    :param tvec: The solvepnp translation vector.
    :return: A tuple containing distance (with the vertical angle compensated for)
     and the angle from the direction the camera is facing to the center of the target.
    """

    rotated = np.matmul(np.squeeze(tvec), tilt_matrix)

    distance = math.sqrt(rotated[0] ** 2 + rotated[2] ** 2)
    angle1 = math.atan2(rotated[0], rotated[2])

    return distance, angle1


out = []


def solvepnp(corners, img):
    distances = []
    angles = []
    tvecs = []
    for target in corners[2:-2]:
        imagepoints = np.zeros((4, 2), dtype=np.float64)
        for index, p in enumerate(target):
            imagepoints[index][0] = p.x
            imagepoints[index][1] = p.y

        # imagepoints = cv2.fisheye.undistortPoints(np.expand_dims(np.asarray(imagepoints), -2), mtx, dist)
        imagepoints = cv2.undistortPoints(imagepoints, mtx, dist)

        success, rotation_vector, translation_vector \
            = cv2.solvePnP(real_coords, imagepoints, np.identity(3), np.zeros(5), flags=0)
        # cv2.aruco.drawAxis(img, mtx, dist, rotation_vector, translation_vector, w / 2)
        # translation_vector[1] *= -1
        # print(translation_vector)

        distance, angle1 = compute_output_values(rotation_vector, translation_vector)
        distances.append(distance)
        angles.append(angle1)
        tvecs.append(translation_vector)

    # cv2.imshow("axis", img)

    if len(distances) == 0:
        return 0.0, 0.0

    x_vals = []
    y_vals = []
    z_vals = []

    for item in tvecs:
        x_vals.append(item[0])
        y_vals.append(item[1])
        z_vals.append(item[2])

    print(np.array([
        statistics.median(x_vals),
        statistics.median(y_vals),
        statistics.median(z_vals)
    ]))

    return statistics.median(distances), statistics.median(angles)
