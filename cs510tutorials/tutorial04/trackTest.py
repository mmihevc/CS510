import cv2
import numpy as np


def frame_track():
    cap = cv2.VideoCapture('IMG_3001.mp4')
    while cap.isOpened():
        # Read video capture
        ret, frame = cap.read()
        # Display each frame
        cv2.imshow("video", frame)
        # show one frame at a time
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        # Quit when 'q' is pressed
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


frame_track()
