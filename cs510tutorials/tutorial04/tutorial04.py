import cv2
import numpy as np


def playDusty():
    cap = cv2.VideoCapture('IMG_3001.mp4')
    while 1:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Image", frame)
            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break

    cv2.destroyAllWindows()
    cap.release()


playDusty()
