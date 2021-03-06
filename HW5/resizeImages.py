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

def getClassName(img_name):
    label = ''
    if "Bighorn_Sheep" in img_name:
        label = "Bighorn_Sheep"
    elif "Bobcat" in img_name:
        label = "Bobcat"
    elif "Coyote" in img_name:
        label = "Coyote"
    elif "Gray_Fox" in img_name:
        label = "Gray_Fox"
    elif "Javelina" in img_name:
        label = "Javelina"
    elif "Mule_Deer" in img_name:
        label = "Mule_Deer"
    elif "Raptor" in img_name:
        label = "Raptor"
    elif "White_tailed_Deer" in img_name:
        label = "White_tailed_Deer"
    else:
        return

    values = img_name.split("_")
    hour = int(values[len(values) - 3])
    chip = "chip" + values[len(values) - 7]
    if hour < 7 or hour >= 18:
            label += "-night-" + str(hour)
    else:
            label += "-day-" + str(hour)

    return f"{label}-{chip}"

#CHANGE THIS
inputFolder = "C:\\Users\\Devin\\Desktop\\ImageComp\\CS510\\cs510bmgr8"
#inputFolder = "/s/chopin/k/grad/mmihevc/CS510/HW5/cs510bmgr8"
if os.path.exists("resizedAnimals"):
    shutil.rmtree("resizedAnimals")
ensure_dir("resizedAnimals\\Test")
ensure_dir("resizedAnimals\\Train")
ensure_dir("resizedAnimals\\Validation")
# ensure_dir("resizedAnimals/Test")
# ensure_dir("resizedAnimals/Train")

paths = glob.glob(inputFolder + "*")
matches = [".jpg", ".JPG"]
testCount = 1
trainCount = 1
validationCount = 1
saveValCount = 1
for class_path in paths:
    i = 0
    #for img_path in glob.glob(class_path + "/*.jpg")[0:5000]:
    for img_path in glob.glob(class_path + "/*.JPG"):
        try:
            image = cv2.imread(img_path)
            imgResized = cv2.resize(image, (256,256))
            label = getClassName(img_path)

            if any(x in label for x in ["chip02", "chip03", "chip04", "chip05", "chip06"]):
                if not cv2.imwrite(f"resizedAnimals/Test/{label}-{str(testCount)}.JPG", imgResized):
                    raise Exception("Could not write image")
                else:
                    testCount += 1
                      
            else:
                if(saveValCount % 10 == 0):
                    if not cv2.imwrite(f"resizedAnimals/Validation/{label}-{str(validationCount)}.JPG", imgResized):
                        raise Exception("Could not write image")
                    else:
                        validationCount += 1
                        saveValCount = 1
                else:
                    if not cv2.imwrite(f"resizedAnimals/Train/{label}-{str(trainCount)}.JPG", imgResized):
                        raise Exception("Could not write image")
                    else:
                        trainCount += 1
                        saveValCount += 1
        
            # cv2.imshow(f"{label}", imgResized)
            # cv2.waitKey(30)
            # cv2.destroyAllWindows()
        except:
            print("An exception occurred")