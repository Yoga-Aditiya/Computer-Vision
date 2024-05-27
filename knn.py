# knn.py

import cv2 as cv
import numpy as np

capture = cv.VideoCapture('data/mouse.mp4')
if not capture.isOpened():
    exit(0)

subsKNN = cv.createBackgroundSubtractorKNN()

while capture.isOpened():
    re, frame = capture.read()
    scale = 20

    if isinstance(frame, type(None)):
        break

    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)
    image = cv.resize(frame, dim, cv.INTER_AREA)
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    image = cv.filter2D(image, -1, gaussian)

    blobKNN = subsKNN.apply(image)

    cv.imshow("image asli", image)
    cv.imshow("image knn", blobKNN)
    keyword = cv.waitKey(30)
    if keyword == 'q' or keyword == 27:
        break
cv.destroyAllWindows()
exit(0)
