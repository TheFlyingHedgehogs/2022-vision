import time
import cv2
from numpy.typing import ArrayLike
import stages


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

while True:
    im: ArrayLike = prov.read()
    # cv2.imshow("input", im)
    contours = stages.find_filter_contours(im)
    corners = stages.find_corners(contours, im)
    distance, angle = stages.solvepnp(corners, im)
    print(f"dst: {distance} ang: {angle}")
    break
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
