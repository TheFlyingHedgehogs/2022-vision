import numpy as np
from math import sin, cos, pi, radians

tvec = np.array([[0.0156668, ],
                 [0.46285442, ],
                 [6.56567066, ]])

tilt_angle = radians(-20)

tilt_matrix = np.array([
    [1, 0, 0],
    [0, cos(-tilt_angle), -sin(-tilt_angle)],
    [0, sin(-tilt_angle), cos(-tilt_angle)]
])

print(np.matmul(np.squeeze(tvec), tilt_matrix))

"""
[[0.0156668 ]
 [0.46285442]
 [6.56567066]]
rotated [ 0.0156668  -1.81065073  6.32801781]
"""
