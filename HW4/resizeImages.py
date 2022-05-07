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

def getClassName(self, img_name):
    label = ''
    if "Bighorn_Sheep" in img_name:
        self.bigHornCount += 1
        label = "Bighorn_Sheep"
    elif "Bobcat" in img_name:
        self.bobcatCount += 1
        label = "Bobcat"
    elif "Coyote" in img_name:
        self.coyoteCount += 1
        label = "Coyote"
    elif "Gray_Fox" in img_name:
        self.grayFoxCount += 1
        label = "Gray_Fox"
    elif "Javelina" in img_name:
        self.javelinaCount += 1
        label = "Javelina"
    elif "Mule_Deer" in img_name:
        self.muleDeerCount += 1
        label = "Mule_Deer"
    elif "Raptor" in img_name:
        self.raptorCount += 1
        label = "Raptor"
    elif "White_tailed_Deer" in img_name:
        self.whiteTailCount += 1
        label = "White_tailed_Deer"
    else:
        return

    values = img_name.split("_")
    hour = int(values[values.length - 3])
    if hour < 7 or hour > 19:
            label += "/day"
    else:
            label += "/night"

    return label

#CHANGE THIS
inputFolder = "C:\Users\Devin\Desktop\ImageComp\CS510\cs510bmgr8"
#inputFolder = "/Users/maddiemihevc/IdeaProjects/CS510/HW4/cs510chips/"
if os.path.exists("resizedAnimals"):
    shutil.rmtree("resizedAnimals")
ensure_dir("resizedAnimals")

paths = glob.glob(inputFolder + "*")
matches = [".jpg", ".JPG"]
for file_path in paths:
        file = file_path.split(os.sep)[-1]
        if any(x in file for x in matches):
            image = cv2.imread(file_path)
            imgResized = cv2.resize(image, (256,256))
            cv2.imwrite(f"resizedAnimals/{file}", imgResized)
            cv2.imshow(f"resizedAnimals", imgResized)
            cv2.waitKey(30)
        else:    
            for img_path in glob.glob(file_path + "/*"):
                    file = img_path.split(os.sep)[-1]
                    if any(x in file for x in matches):
                            image = cv2.imread(img_path)
                            imgResized = cv2.resize(image, (256,256))
                            cv2.imwrite(f"resizedAnimals/{file}", imgResized)
                            cv2.imshow(f"resizedAnimals", imgResized)
                            cv2.waitKey(30)
                            
cv2.destroyAllWindows()