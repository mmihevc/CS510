import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

threshold = 0.8
m = 'cv.TM_CCOEFF_NORMED'

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
           'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
colorIndex = 0

templates = []
recognition = False

class Template:
    def __init__(self, tempImage, width, height, color, name):
        self.tempImage = tempImage
        self.width = width
        self.height = height
        self.color = color
        self.name = name

    tempImage: any
    width: any
    height: any
    color: any
    name: any

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 1:
        includeCapDShow = int(args[0])
    if len(args) >= 2:
        threshold = float(args[1])
    if len(args) >= 3:
        m = str(args[2])

    # Initialize the webcam
    if includeCapDShow == 0:
        video = cv.VideoCapture(0)
    if includeCapDShow == 1:
        video = cv.VideoCapture(0, cv.CAP_DSHOW)

    # Exit if video not opened.
    if not video.isOpened():
        print("Failed to open the video\n")
        sys.exit()

    while True:
        # Read each frame from the video
        ret, frame = video.read()
        if not ret:
            continue

        # Flip the frame vertically
        frame = cv.flip(frame, 1)
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if recognition:
            header = "Recognition Mode ON - threshold:" + str(float(threshold))
            #for m in methods: (I think we should pass in the method from the command line)
            for t in templates:
                method = eval(m)

                res = cv.matchTemplate(frame, t.tempImage, method)
                min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

                if max_val > threshold:
                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                        top_left = min_loc
                    else:
                        top_left = max_loc
                    bottom_right = (top_left[0] + t.width, top_left[1] + t.height)

                     # putting  rectangle on recognized area
                    cv.rectangle(frame, top_left, bottom_right, t.color, 2)

                    formattedMaxVal = "{:.3f}".format(max_val)
                    text_width, text_height = cv.getTextSize(t.name + " - " + formattedMaxVal, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)    
                    cv.putText(frame, t.name + " - " + formattedMaxVal, (top_left[0], top_left[1] + t.height + (text_height * 3)), cv.FONT_HERSHEY_SIMPLEX, 0.5, t.color, 2)
        else:
            header = "Recognition Mode OFF"

            
        cv.imshow(header, frame)

        k = cv.waitKey(30)
        if k & 0xFF == ord('b'):
            name = input("Enter object name:")

            bbox = cv.selectROI(frame, False)
            tempImage = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
            w = tempImage.shape[1]
            h = tempImage.shape[0]

            template = Template(tempImage, w, h, colors[colorIndex], name)
            colorIndex += 1
            
            #wrap the color list back to the start
            if colorIndex == 5:
                colorIndex = 0
            
            templates.append(template)
            print("Templated Captured")
        elif k & 0xFF == ord('r'):
            recognition = True
            print("Recognition Mode On")
        elif k & 0xFF == ord('u'):
            recognition = False
            print("Recognition Mode Off")
        elif k & 0xFF == ord('l'):
            print("Clearing Templates, Recognition Mode Off")
            recognition = False
            colorIndex = 0
            templates = []
        if k & 0xFF == ord('q'):
            break

            # release the webcam and destroy all active windows
    video.release()
    cv.destroyAllWindows()
