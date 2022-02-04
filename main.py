import math
import statistics
import time
import cv2
from numpy.typing import ArrayLike
import stages
import utils
from angles import find_pose
import networktables


class ImageProv:
    def read(self) -> ArrayLike:
        return None


class ImageRead(ImageProv):
    def __init__(self, filename: str):
        self.img = cv2.imread(filename)

    def read(self) -> ArrayLike:
        return self.img.copy()


class VideoCap(ImageProv):
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.image = None

    def read(self) -> ArrayLike:
        return self.cap.read()[1]
        # r = self.cap.read(self.image)
        # if self.image is not None:
        #     return self.image
        # else:
        #     self.image = np.zeros(r[1].shape)
        #     return r[1]


cam = cv2.VideoCapture(0)
# cam.open(2, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

prov: ImageProv = VideoCap(cam)
start = time.monotonic()
framecount = 0
total = 0
#table = networktables.NetworkTablesInstance.getDefault().getTable("vision")
#nt = networktables.NetworkTablesInstance.getDefault()
#nt.setServer("10.28.98.2")
#table = nt.getTable("vision")
#table = networktables.NetworkTables.initialize("10.28.98.2").getTable("vision")
networktables.NetworkTables.initialize("10.28.98.2")
table = networktables.NetworkTables.getTable("vision")

distance_entry = table.getEntry("distance")
angle_entry = table.getEntry("angle")
# for i in range(200):
avg_d = []
avg_a = []
window = 20
while True:
    im: ArrayLike = prov.read()
    # im2 = im.copy()
    # cv2.drawMarker(im2, (500, 500), (255, 255, 0))

    # cv2.imshow("input", im2)
    # timeit("contours", True)
    contours = stages.find_filter_contours(im)
    # timeit("contours")
    # timeit("corners", True)
    corners = stages.find_corners(contours, im)
    # timeit("corners")
    # timeit("solvepnp", True)
    distance, angle = stages.solvepnp(corners, im)
    # timeit("solvepnp")
    #break

    # undistorted = cv2.fisheye.undistortImage(im, stages.mtx, stages.dist)
    # h, w = 1080, 1920
    # K = stages.mtx
    # D = stages.dist
    # DIM = np.array([1920, 1080])
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # undistorted = cv2.remap(im, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.

    # newpos = cv2.fisheye.undistortPoints(np.array([[[500.0, 500.0]]], dtype=np.float32), K, D)
    # newx = map1[500][500][0]
    # newy = map2[500][500]
    # print(newx)
    # print(newy)
    # print(newx, newy)
    # cv2.drawMarker(undistorted, (int(newx), int(newy)), (255, 255, 0))
    # print(newpos)
    # print(map1)
    # print(map2)

    # cv2.imshow("undistorted", undistorted)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # print(".", end="")
    framecount += 1
    avg_d.append(distance)
    avg_a.append(angle)
    if len(avg_d) >= window:
        avg_d.pop(0)
        avg_a.pop(0)

    #if distance == 0 and angle == 0:
    #    continue

    distance_entry.setDouble(distance)
    angle_entry.setDouble(angle)

    # if framecount == 25:
    #     print(f"dst: {distance} ang: {angle}")
    #     dist = statistics.mean(avg_d)
    #     ang = statistics.mean(avg_a)
    #     print(f"avg dist: {dist} avg ang: {ang}")
    #     x, y = find_pose(dist, math.radians(ang), math.radians(0))
    #     print(f"x: {x} y: {y}")
    #     avg = (time.monotonic() - start) / framecount
    #     framecount = 0
    #     print(f"avg framerate: {1 / avg}")
    #     start = time.monotonic()

# cv2.destroyAllWindows()

# print(utils.timing)
