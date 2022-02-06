import cv2
import numpy as np

img = cv2.imread('IconFaceLv2.png')

def showPanda() :
    # cv2.imshow('image',img)
    cv2.namedWindow('panda',cv2.WINDOW_NORMAL)
    cv2.imshow('panda',img)
    cv2.waitKey(0)
    
