import cv2
import numpy as np

img = cv2.imread('CalTech_256_084_0082.jpg')

def warpGiraffe() :
    cv2.namedWindow('imgsrc',cv2.WINDOW_NORMAL)
    cv2.namedWindow('imgdst',cv2.WINDOW_NORMAL)
    rows,cols,chans = img.shape
    
    pts1 = np.float32([[0,0],[0,300],[241,0]])
    pts2 = np.float32([[30,30],[0,300],[241,0]])
    M    = cv2.getAffineTransform(pts1,pts2)
    print(M)
    dst  = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow('imgsrc',img)
    cv2.imshow('imgdst',dst)
    cv2.waitKey(0)

if __name__ == '__main__':
        warpGiraffe()
