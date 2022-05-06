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

    utils.timeit("hsv", True)
    hsv_cache = cv2.cvtColor(img, cv2.COLOR_RGB2HSV, hsv_cache)
    utils.timeit("hsv")
    utils.timeit("inRange", True)
    thresh_cache = cv2.inRange(hsv_cache, (30, 100, 80), (60, 255, 255), thresh_cache)
    utils.timeit("inRange")
    utils.timeit("contours", True)
    contours, _ = cv2.findContours(thresh_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    utils.timeit("contours")
    utils.timeit("filter", True)
    output = []
    for c in contours:
        area = cv2.contourArea(c)
        # if area < 75:
        #    continue
        output.append(c)
    utils.timeit("filter")
    if utils.DISPLAY:
        im2 = img.copy()
        cv2.drawContours(im2, output, -1, (255, 0, 0))
        cv2.imshow("contours", im2)
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


def wrapping_slice(arr, start, end, t):
    if end < start:
        start -= t
        end += t
    else:
        start += t
        end -= t

    end %= len(arr)
    start %= len(arr)

    if end < start or end < 0 or start < 0:
        return np.array(np.ndarray.tolist(arr[:end]) + np.ndarray.tolist(arr[start:]))
    else:
        return arr[start:end]


# no questions allowed
def find_corners_2(corners, contour):
    def find_corner(c):
        corn = -1
        value = 10000000
        for index, item in enumerate(contour):
            dst = abs(item[0] - c.x) + abs(item[1] - c.y)
            if dst < value:
                corn = index
                value = dst
        return corn

    corner1 = find_corner(corners[0])
    corner2 = find_corner(corners[1])
    corner3 = find_corner(corners[2])
    corner4 = find_corner(corners[3])

    segment1 = wrapping_slice(contour, corner1, corner2, 2)
    segment2 = wrapping_slice(contour, corner2, corner3, 2)
    segment3 = wrapping_slice(contour, corner3, corner4, 2)
    segment4 = wrapping_slice(contour, corner4, corner1, 2)

    if len(segment1) < 2 or len(segment2) < 2 or len(segment3) < 2 or len(segment4) < 2:
        return corners

    def fit(a, b):
        if len(set(a)) < 2:
            return np.array([1000, a[0] * -1000])
        return np.polyfit(a, b, deg=1)

    regression1 = fit(segment1[:, 0], segment1[:, 1])
    regression2 = fit(segment2[:, 0], segment2[:, 1])
    regression3 = fit(segment3[:, 0], segment3[:, 1])
    regression4 = fit(segment4[:, 0], segment4[:, 1])

    def solve(a, b):
        m1 = a[0]
        m2 = b[0]
        b1 = a[1]
        b2 = b[1]
        x = (b2 - b1) / (m1 - m2)
        return Point(x, m1 * x + b1)

    c1 = solve(regression1, regression4)
    c2 = solve(regression1, regression2)
    c3 = solve(regression2, regression3)
    c4 = solve(regression3, regression4)

    def dst(a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)

    t = 10
    if dst(c1, corners[0]) > t or dst(c2, corners[1]) > t or dst(c3, corners[2]) > t or dst(c4, corners[3]) > t:
        return None

    return [c1, c2, c3, c4]


def find_corners(contours, img):
    corners = []
    for contour in contours:
        contour = np.squeeze(contour)
        if len(contour) < 4:
            continue
        #        for i in range(0, len(contour)):
        #            contour[i][0] = 1920 - contour[i][0]
        #            contour[i][1] = 1080 - contour[i][1]
        rotated = np.matmul(contour, rotation_matrix)
        leftbottom_rot = rotated[0]
        righttop_rot = rotated[0]
        lefttop_rot = rotated[0]
        rightbottom_rot = rotated[0]
        for item in rotated:
            if isinstance(item, float):
                continue
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

        c = [
            Point(round(lb[0]), round(lb[1])),
            Point(round(rb[0]), round(rb[1])),
            Point(round(rt[0]), round(rt[1])),
            Point(round(lt[0]), round(lt[1])),
        ]
        # corners.append(c)
        res = find_corners_2(c, contour)
        if res:
            corners.append(res)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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

    # for item in refined:
    #    item[0] = 1920 - item[0]
    #    item[1] = 1080 - item[1]
    # refined = np.matmul(refined, invert_matrix)
    # for item in refined:
    #    item[0] += 1920 / 2
    #    item[1] += 1080 / 2

    output = corners

    # leaving this commented out, it's slow and the window it makes is huge
    img2 = cv2.resize(img, (1920 * 4, 1080 * 4), interpolation=cv2.INTER_NEAREST)
    # #img2 = img.copy()
    # for quad in output:
    #     print(quad)
    #     cv2.drawMarker(img2, (int(quad[0].x / (1920 / 500)), int(quad[0].y / (1080 / 500))), (0, 0, 255))
    #     cv2.drawMarker(img2, (int(quad[1].x / (1920 / 500)), int(quad[1].y / (1080 / 500))), (0, 255, 0))
    #     cv2.drawMarker(img2, (int(quad[2].x / (1920 / 500)), int(quad[2].y / (1080 / 500))), (0, 255, 255))
    #     cv2.drawMarker(img2, (int(quad[3].x / (1920 / 500)), int(quad[3].y / (1080 / 500))), (255, 0, 0))
    for quad in corners:
        cv2.drawMarker(img2, (int(quad[0].x * 4), int(quad[0].y * 4)), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS)
        cv2.drawMarker(img2, (int(quad[1].x * 4), int(quad[1].y * 4)), (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS)
        cv2.drawMarker(img2, (int(quad[2].x * 4), int(quad[2].y * 4)), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS)
        cv2.drawMarker(img2, (int(quad[3].x * 4), int(quad[3].y * 4)), (255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS)
    cv2.imshow("corners", img2)

    return output


w = 0.063405 * 2
h = 2 * 2.54 / 100

real_coords = np.array([
    [0, 0, 0],  # left bottom
    [w, 0, 0],  # right bottom
    [w, h, 0],  # right top
    [0, h, 0]  # left top
], dtype="double")

for triple in real_coords:
    triple[0] -= w / 2
    triple[2] += 0.674882

# /home/pi/2898-2022-vision-py/
# mtx, dist, rvecs, tvecs = pkl.load(open("calib/picam-2/calib.pkl", "rb"))
# mtx, dist, rvecs, tvecs = pkl.load(open("/home/pi/2898-2022-vision-py/calib/picam-2/calib.pkl", "rb"))
# mtx, dist, rvecs, tvecs = pkl.load(open("calib/virtual-camera-2/calib.pkl", "rb"))
mtx, dist = pkl.load(open("/home/max/synced/vision-testing/calib/calib.pkl", "rb"))
# print(dist)

tilt_angle = math.radians(-20)

# x
tilt_matrix = np.array([
    [1, 0, 0],
    [0, cos(-tilt_angle), -sin(-tilt_angle)],
    [0, sin(-tilt_angle), cos(-tilt_angle)]
])

# # y
# tilt_matrix = np.array([
#     [cos(-tilt_angle), 0, sin(-tilt_angle)],
#     [0, 1, 0],
#     [-sin(-tilt_angle), 0, cos(-tilt_angle)]
# ])

# # z
# tilt_matrix = np.array([
#     [cos(-tilt_angle), -sin(-tilt_angle), 0],
#     [sin(-tilt_angle), cos(tilt_angle), 0],
#     [0, 0, 1]
# ])

"""
z pos 6.780974423515335 neg 6.709745617917127
y pos 6.747855029228697 neg 6.747855029228697
x pos 6.104506452572762 neg 6.617734721310734

aaa
z pos 7.079253621741147 neg 7.1586818131666545
y pos 7.130354067016877 neg 7.130354067016876
x pos 6.93618229082667 neg 6.497784797552637
"""


def compute_output_values(rvec, tvec):
    """
    Computes the distance and angle to the target.

    :param rvec: The solvepnp rotation vector, unused as of now.
    :param tvec: The solvepnp translation vector.
    :return: A tuple containing distance (with the vertical angle compensated for)
     and the angle from the direction the camera is facing to the center of the target.
    """

    # tvec = np.array([1.5, 0.37234, 6.3851])

    # tvec format:
    # 0 - x (side to side from the camera's perspective)
    # 1 - y (up and down from the camera)
    # 2 - z (in and out from the camera)
    tx = tvec[0][0]
    ty = tvec[1][0]
    tz = tvec[2][0]

    p1 = (0.0, sin(tilt_angle) * tz, cos(tilt_angle) * tz)
    a1 = pi / 2 + tilt_angle
    p2 = (tx, p1[1] + ty * sin(a1), p1[2] + ty * cos(a1))
    rotated = p2

    distance = math.sqrt(rotated[0] ** 2 + rotated[2] ** 2)
    angle1 = math.atan2(rotated[0], rotated[2])
    # print(rotated)
    # print(tvec)

    return distance, angle1


out = []


def solvepnp(corners, img):
    distances = []
    angles = []
    tvecs = []
    for target in corners:  # [2:-2]:
        imagepoints = np.zeros((4, 2), dtype=np.float64)
        for index, p in enumerate(target):
            imagepoints[index][0] = p.x
            imagepoints[index][1] = p.y

        # imagepoints = cv2.fisheye.undistortPoints(np.expand_dims(np.asarray(imagepoints), -2), mtx, dist)
        # imagepoints = cv2.undistortPoints(imagepoints, mtx, dist)

        # success, rotation_vector, translation_vector \
        #     = cv2.solvePnP(real_coords, imagepoints, np.identity(3), np.zeros(5), flags=0)
        success, rotation_vector, translation_vector \
            = cv2.solvePnP(real_coords, imagepoints, mtx, dist, flags=0)
        # print(translation_vector)
        if utils.DISPLAY:
            cv2.aruco.drawAxis(img, mtx, dist, rotation_vector, translation_vector, w / 2)
        # translation_vector[1] *= -1
        # print(translation_vector)

        distance, angle1 = compute_output_values(rotation_vector, translation_vector)
        distances.append(distance)
        angles.append(angle1)
        tvecs.append(translation_vector)

    if utils.DISPLAY:
        cv2.imshow("axis", img)

    if len(distances) == 0:
        return 0.0, 0.0

    # print(f"{statistics.median(map(lambda x: x[0][0], tvecs))}, {statistics.median(map(lambda x: x[1][0], tvecs))}, {statistics.median(map(lambda x: x[2][0], tvecs))}")

    x_vals = []
    y_vals = []
    z_vals = []

    for item in tvecs:
        x_vals.append(item[0])
        y_vals.append(item[1])
        z_vals.append(item[2])

    return statistics.median(distances), statistics.median(angles)
