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


prov: ImageProv = ImageRead("images/test1.png")

while True:
    im: ArrayLike = prov.read()
    cv2.imshow("input", im)
    stages.find_filter_contours(im)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
