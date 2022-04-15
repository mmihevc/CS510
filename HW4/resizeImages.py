"""
Script to help generate images of the same size for the training, change the input folder and output location,
this code will recursively loop through the folders and create 1024 images for each genre (this value can also be changed)
"""

import cv2
import glob
import os
import shutil

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#CHANGE THIS
inputFolder = "C:\\Users\\Devin\\Desktop\\GitHub\\cs510chips\\"
if os.path.exists("resizedAnimals"):
    shutil.rmtree("resizedAnimals")
ensure_dir("resizedAnimals")

paths = glob.glob(inputFolder + "*")
matches = [".jpg", ".JPG"]
for file_path in paths:
        file = file_path.split("\\")[-1]
        if any(x in file for x in matches):
            image = cv2.imread(file_path)
            imgResized = cv2.resize(image, (64,64))
            cv2.imwrite(f"resizedAnimals/{file}.jpg", imgResized)
            cv2.imshow(f"resizedAnimals", imgResized)
            cv2.waitKey(30)
        else:    
            for img_path in glob.glob(file_path + "/*.jpg"):
                    file = img_path.split("\\")[-1]
                    if any(x in file for x in matches):
                            image = cv2.imread(img_path)
                            imgResized = cv2.resize(image, (64,64))
                            cv2.imwrite(f"resizedAnimals/{file}.jpg", imgResized)
                            cv2.imshow(f"resizedAnimals", imgResized)
                            cv2.waitKey(30)
                            
cv2.destroyAllWindows()