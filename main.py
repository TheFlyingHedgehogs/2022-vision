import math
import statistics
import time
import cv2
import numpy as np
import stages
import utils
from angles import find_pose
import networktables
from utils import timeit
import picamera


class ImageProv:
    def read(self):
        return None


class ImageRead(ImageProv):
    def __init__(self, filename: str):
        self.img = cv2.imread(filename)

    def read(self):
        return self.img.copy()


class VideoCap(ImageProv):
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.image = None

    def read(self):
        return self.cap.read()[1]
        # r = self.cap.read(self.image)
        # if self.image is not None:
        #     return self.image
        # else:
        #     self.image = np.zeros(r[1].shape)
        #     return r[1]


class PiCamCap(ImageProv):
    def __init__(self):
        self.cam = picamera.PiCamera()
        self.cam.__enter__()
        camera = self.cam
        camera.resolution = (1920, 1080)
        camera.framerate = 24
        camera.awb_mode = "sunlight"
        camera.exposure_mode = "off"
        camera.shutter_speed = 2000
        time.sleep(2)
        self.image = np.empty((1088, 1920, 3), dtype=np.uint8)
        self.it = iter(camera.capture_continuous(self.image, use_video_port=True, format="bgr"))

    def read(self):
        #self.cam.capture(self.image, "bgr")
        #return self.image[:1080, :1920, :]
        next(self.it)
        return self.image


prov: ImageProv = PiCamCap()
start = time.monotonic()
framecount = 0
total = 0
if utils.NETWORKTABLES:
    networktables.NetworkTables.initialize("10.28.98.2")
    table = networktables.NetworkTables.getTable("vision")

    distance_entry = table.getEntry("distance")
    angle_entry = table.getEntry("angle")
avg_d = []
avg_a = []
window = 50
while True:
    im = prov.read()

    timeit("contours", True)
    contours = stages.find_filter_contours(im)
    timeit("contours")

    timeit("corners", True)
    corners = stages.find_corners(contours, im)
    timeit("corners")

    timeit("solvepnp", True)
    distance, angle = stages.solvepnp(corners, im)
    timeit("solvepnp")

    if utils.DISPLAY and cv2.waitKey(100) & 0xFF == ord('q'):
        break

    framecount += 1
    avg_d.append(distance)
    avg_a.append(angle)
    if len(avg_d) >= window:
        avg_d.pop(0)
        avg_a.pop(0)

    if utils.NETWORKTABLES and distance != 0 and angle != 0:
        distance_entry.setDouble(distance)
        angle_entry.setDouble(angle)

    if framecount == 25:
        print(f"dst: {distance} ang: {angle}")
        dist = statistics.mean(avg_d)
        ang = statistics.mean(avg_a)
        print(f"avg dist: {dist} avg ang: {ang}")
        x, y = find_pose(dist, math.radians(ang), math.radians(0))
        print(f"x: {x} y: {y}")
        avg = (time.monotonic() - start) / framecount
        framecount = 0
        print(f"avg framerate: {1 / avg}")
        start = time.monotonic()

if utils.DISPLAY:
    cv2.destroyAllWindows()

if utils.BENCHMARK:
    print(utils.timing)
