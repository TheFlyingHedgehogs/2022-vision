# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

res = (1920, 1080)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = res
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=res)
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array
    # show the frame
    cv2.imshow("Frame", cv2.resize(image, (100, 100)))
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
