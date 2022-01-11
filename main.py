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


prov: ImageProv = ImageRead("images/20m-0d-15d-tilt.png")

while True:
    im: ArrayLike = prov.read()
    cv2.imshow("input", im)
    contours = stages.find_filter_contours(im)
    corners = stages.find_corners(contours, im)
    stages.solvepnp(corners, im)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
