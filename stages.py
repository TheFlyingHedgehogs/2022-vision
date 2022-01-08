from math import cos, pi, sin
import cv2
import numpy as np
import dataclasses
from numpy.typing import ArrayLike

thresh_cache = None
hsv_cache = None


def find_filter_contours(img: ArrayLike) -> list[ArrayLike]:
    global thresh_cache, hsv_cache

    hsv_cache = cv2.cvtColor(img, cv2.COLOR_RGB2HSV, hsv_cache)
    thresh_cache = cv2.inRange(hsv_cache, (25, 10, 10), (100, 255, 255), thresh_cache)
    contours, _ = cv2.findContours(thresh_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0))
    cv2.imshow("contours", img)
    return contours


@dataclasses.dataclass
class Point:
    x: int
    y: int


theta = pi / 2
rotation_matrix = np.array([
    [cos(theta), -sin(theta), 0],
    [sin(theta), cos(theta), 0],
    [0, 0, 1]
])


def find_corners(contours: list[ArrayLike]) -> list[Point]:
    corners: list[list[Point]] = []
    for contour in contours:
