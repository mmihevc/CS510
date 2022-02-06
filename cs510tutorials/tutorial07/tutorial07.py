import cv2
import numpy as np

img = cv2.imread('CalTech_256_084_0082.jpg')

def warpGiraffe() :
    cv2.namedWindow('imgsrc',cv2.WINDOW_NORMAL)
    cv2.namedWindow('imgdst',cv2.WINDOW_NORMAL)
    rows,cols,chans = img.shape
    
    pts1 = np.float32([[0,0],[0,300],[241,0],[241,300]])
    #pts2 = np.float32([[0,0],[0,300],[200,50],[200,250]])
    pts2 = np.float32([[50,0],[0,300],[350,0],[241,300]])
    M    = cv2.getPerspectiveTransform(pts1,pts2)
    print(M)
    dst  = cv2.warpPerspective(img,M,(cols,rows))
    cv2.imshow('imgsrc',img)
    cv2.imshow('imgdst',dst)
    cv2.waitKey(0)

if __name__ == '__main__':
    warpGiraffe()
