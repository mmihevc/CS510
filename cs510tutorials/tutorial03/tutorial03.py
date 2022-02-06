import cv2
import numpy as np

im = cv2.imread('IconFaceLv2.png')

def colorPanda() :
    # Example extended from
    # https://www.learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
     
    # Select ROI
    r = cv2.selectROI(im)
     
    # Turn pixels in the ROI Red !
    im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = [0,0,255]
 
    # Display altered image
    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
