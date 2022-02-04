import math


def find_pose(distance_m: float, angle_rad: float, absolute_angle_rad: float):
    c = (math.pi / 2) - (absolute_angle_rad + angle_rad)
    x = math.cos(c) * distance_m
    y = math.sin(c) * distance_m
    return x, y
