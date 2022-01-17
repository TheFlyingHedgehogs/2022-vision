import time
import cv2
from numpy.typing import ArrayLike
import stages
import utils
from utils import timeit


class ImageProv:
    def read(self) -> ArrayLike:
        return None


class ImageRead(ImageProv):
    def __init__(self, filename: str):
        self.img = cv2.imread(filename)

    def read(self) -> ArrayLike:
        return self.img.copy()


prov: ImageProv = ImageRead("images/20d-20d-up.png")
start = time.monotonic()
framecount = 0
total = 0

for i in range(200):
    im: ArrayLike = prov.read()
    # cv2.imshow("input", im)
    timeit("contours", True)
    contours = stages.find_filter_contours(im)
    timeit("contours")
    timeit("corners", True)
    corners = stages.find_corners(contours, im)
    timeit("corners")
    timeit("solvepnp", True)
    distance, angle = stages.solvepnp(corners, im)
    timeit("solvepnp")
    #print(f"dst: {distance} ang: {angle}")
    #break
    # if cv2.waitKey(100) & 0xFF == ord('q'):
    #     break
    # print(".", end="")
    framecount += 1
    if framecount == 100:
        avg = (time.monotonic() - start) / framecount
        framecount = 0
        print(f"avg framerate: {1 / avg}")
        start = time.monotonic()

# cv2.destroyAllWindows()

print(utils.timing)
