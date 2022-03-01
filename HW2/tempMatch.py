import sys
import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

if __name__ == '__main__' :
    args = sys.argv[1:]
    if len(args) >= 1:
        threshold = float(args[0])

    # Initialize the webcam
    video = cv.VideoCapture(0, cv.CAP_DSHOW) 
    recognition = False

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
   

        k = cv.waitKey(1)
        if(k & 0xFF == ord('b')): 
            bbox = cv.selectROI(frame, False)
            tempImage = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] 
            w = tempImage.shape[1]
            h = tempImage.shape[0]
        elif(k & 0xFF == ord('r')): 
            recognition = True
            print("in recognition mode")

        if (recognition):
            res = cv.matchTemplate(frame,tempImage,cv.TM_CCOEFF_NORMED)
            location = np.where( res >= threshold)
            for pt in zip(*location[::-1]):
                #puting  rectangle on recognized erea 
                cv.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)  
        
        cv.imshow("Output", frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break 

     # release the webcam and destroy all active windows
    video.release()
    cv.destroyAllWindows()
