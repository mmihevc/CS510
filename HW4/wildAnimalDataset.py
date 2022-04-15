"""
Sources: examples from this collection: https://github.com/aladdinpersson/Machine-Learning-Collection
"""

import os
import pandas as pd
import torch
import skimage
from torch.utils.data import Dataset
from skimage import io
import glob
import cv2

class WildAnimals(Dataset):
    def __init__(self, transform=None):
        self.bigHornCount = 0
        self.bobcatCount = 0
        self.coyoteCount = 0
        self.grayFoxCount = 0
        self.javelinaCount = 0
        self.muleDeerCount = 0
        self.raptorCount = 0
        self.whiteTailCount = 0
        self.imgs_path = "resizedAnimals\\"
        self.transform = transform
        file_list = glob.glob(self.imgs_path + "*")

        self.data = []
        matches = [".jpg", ".JPG"]
        for file_path in file_list:
            file = file_path.split("\\")[-1]
            if any(x in file for x in matches):
                class_name = self.getClassName(file)
                self.data.append([file_path, class_name])
            else:    
                for img_path in glob.glob(file_path + "/*.jpg"):
                        file = img_path.split("\\")[-1]
                        if any(x in file for x in matches):
                                class_name = self.getClassName(file)
                                self.data.append([img_path, class_name])

        print("Bighorn_Sheep:" + str(self.bigHornCount))
        print("Bobcat:" +  str(self.bobcatCount))
        print("Coyote:" +  str(self.coyoteCount))
        print("Gray_Fox:" +  str(self.grayFoxCount))
        print("Javelina:" +  str(self.javelinaCount))
        print("Mule_Deer:" +  str(self.muleDeerCount))
        print("Raptor:" +  str(self.raptorCount))
        print("White_tailed_Deer:" +  str(self.whiteTailCount))

        self.class_map = {
        "Bighorn_Sheep" : 0,
        "Bobcat": 1,
        "Coyote": 2,
        "Gray_Fox": 3,
        "Javelina": 4,
        "Mule_Deer": 5,
        "Raptor": 6,
        "White_tailed_Deer": 7
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, class_name = self.data[index]
        image = io.imread(img_path)
        y_label = torch.tensor(self.class_map[class_name])

        if(self.transform):
            image = self.transform(image)
        
        return (image, y_label)
    
    def getClassName(self, img_name):
        if "Bighorn_Sheep" in img_name:
            self.bigHornCount += 1
            return "Bighorn_Sheep"
        elif "Bobcat" in img_name:
            self.bobcatCount += 1
            return "Bobcat"
        elif "Coyote" in img_name:
            self.coyoteCount += 1
            return "Coyote"
        elif "Gray_Fox" in img_name:
            self.grayFoxCount += 1
            return "Gray_Fox"
        elif "Javelina" in img_name:
            self.javelinaCount += 1
            return "Javelina"
        elif "Mule_Deer" in img_name:
            self.muleDeerCount += 1
            return "Mule_Deer"
        elif "Raptor" in img_name:
            self.raptorCount += 1
            return "Raptor"
        elif "White_tailed_Deer" in img_name:
            self.whiteTailCount += 1
            return "White_tailed_Deer"
        else:
            return "None"   

    # todo look into other ways to split the datasets
    def getDatasets(self):
        train_size = int(0.8 * len(self.data))
        test_size = len(self.data) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.data, [train_size, test_size])
        return train_dataset, test_dataset

