import math
import statistics
from math import cos, pi, sin
import cv2
import numpy as np
import dataclasses
from numpy.typing import ArrayLike
import pickle as pkl
import utils

thresh_cache = None
hsv_cache = None


def find_filter_contours(img: ArrayLike) -> list[ArrayLike]:
    global thresh_cache, hsv_cache

    # utils.timeit("hsv", True)
    hsv_cache = cv2.cvtColor(img, cv2.COLOR_RGB2HSV, hsv_cache)
    # utils.timeit("hsv")
    # utils.timeit("inRange", True)
    thresh_cache = cv2.inRange(hsv_cache, (0, 10, 10), (100, 255, 255), thresh_cache)
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


def find_corners(contours: list[ArrayLike], img: ArrayLike) -> list[list[Point]]:
    corners: list[list[Point]] = []
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

        # cv2.drawMarker(img, (int(lb[0]), int(lb[1])), (255, 255, 0))
        # cv2.drawMarker(img, (int(rb[0]), int(rb[1])), (255, 0, 255))
        # cv2.drawMarker(img, (int(lt[0]), int(lt[1])), (255, 255, 255))
        # cv2.drawMarker(img, (int(rt[0]), int(rt[1])), (0, 255, 0))
        #cv2.imshow("corner", img)
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

    refined = cv2.cornerSubPix(gray, converted_corners, (6, 6), (-1, -1), criteria)

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

    # img2 = cv2.resize(img, (1920 * 4, 1080 * 4))

    # for quad in output:
    #     cv2.drawMarker(img2, (int(quad[0].x * 4), int(quad[0].y * 4)), (0, 0, 255))
    #     cv2.drawMarker(img2, (int(quad[1].x * 4), int(quad[1].y * 4)), (0, 255, 0))
    #     cv2.drawMarker(img2, (int(quad[2].x * 4), int(quad[2].y * 4)), (0, 255, 255))
    #     cv2.drawMarker(img2, (int(quad[3].x * 4), int(quad[3].y * 4)), (255, 0, 0))
    # cv2.imshow("img", img)
    # cv2.imshow("corners", img2)

    return output


w = 0.12682
# w = 5 * 2.54 / 100
h = 2 * 2.54 / 100

real_coords = np.array([
    [0, 0, 0],  # left bottom
    [w, 0, 0],  # right bottom
    [w, h, 0],  # right top
    [0, h, 0]   # left top
], dtype="double")

for triple in real_coords:
    triple[0] -= w / 2
    triple[2] += 0.67704

# /home/pi/2898-2022-vision-py/
# mtx, dist, rvecs, tvecs = pkl.load(open("calib/picam-1/calib.pkl", "rb"))
mtx, dist, rvecs, tvecs = pkl.load(open("calib/virtual-camera-2/calib.pkl", "rb"))

# tilt_angle = math.radians(9.33)
tilt_angle = math.radians(-20)
# tilt_angle = math.radians(16.5)

# x
# [ 0.01248718 -1.81977671  6.14176873]
# [2.98642930e-03 2.57007537e+00 5.90170058e+00]
tilt_matrix = np.array([
    [1, 0, 0],
    [0, cos(-tilt_angle), -sin(-tilt_angle)],
    [0, sin(-tilt_angle), cos(-tilt_angle)]
])

# y
# [2.21303927 0.41532774 6.08910597]
# [-2.21871544  0.41532774  6.08704001]
# tilt_matrix = np.array([
#     [cos(-tilt_angle), 0, sin(-tilt_angle)],
#     [0, 1, 0],
#     [-sin(-tilt_angle), 0, cos(-tilt_angle)]
# ])

# z
# [-0.12185139  0.37129401  6.39377504]
# [0.12430388 0.37350121 6.44699508]
# tilt_matrix = np.array([
#     [cos(-tilt_angle), -sin(-tilt_angle), 0],
#     [sin(-tilt_angle), cos(-tilt_angle), 0],
#     [0, 0, 1]
# ])


def compute_output_values(rvec, tvec):  # stolen from ligerbots code
    """Compute the necessary output distance and angles"""

    x = tvec[2]
    y = tvec[0]
    z = tvec[1]

    # hyp = math.sqrt(x ** 2 + z ** 2)
    # ang = math.atan(z / x) + tilt_angle
    #
    # d = hyp / math.cos(ang)
    #
    # distance = math.sqrt(d ** 2 + y ** 2)

    # p1 = [cos(tilt_angle) * x, y, sin(tilt_angle) * x]
    # p1[0] -= sin(tilt_angle) * z
    # p1[2] += cos(tilt_angle) * z
    #
    # distance = math.sqrt(p1[0] ** 2 + p1[1] ** 2)
    # angle1 = 0.0

    rotated = np.matmul(np.squeeze(tvec), tilt_matrix)

    print("AAAAAAAAAAAAAAAA")
    print(rotated)

    distance = 0
    angle1 = 0

    return distance, angle1


def solvepnp(corners: list[list[Point]], img: ArrayLike):
    distances = []
    angles = []
    for target in corners:
        imagepoints = np.zeros((4, 2), dtype=np.float64)
        for index, p in enumerate(target):
            imagepoints[index][0] = p.x
            imagepoints[index][1] = p.y

        # imagepoints = cv2.fisheye.undistortPoints(np.expand_dims(np.asarray(imagepoints), -2), mtx, dist)
        imagepoints = cv2.undistortPoints(imagepoints, mtx, dist)

        success, rotation_vector, translation_vector \
            = cv2.solvePnP(real_coords, imagepoints, np.identity(3), np.zeros(5), flags=0)
        # cv2.aruco.drawAxis(img, mtx, dist, rotation_vector, translation_vector, 0.1)
        translation_vector[1] *= -1
        print(translation_vector)
        distance, angle1 = compute_output_values(rotation_vector, translation_vector)
        distances.append(distance)
        angles.append(angle1)

    # cv2.imshow("axis", img)

    if len(distances) == 0:
        return 0.0, 0.0

    return statistics.median(distances), statistics.median(angles)

    # print(f"distance: {statistics.median(distances)}, angle: {statistics.median(angles)}")
