import os
from turtle import color
import pandas as pd
import torch
import skimage
from torch.utils.data import Dataset
from skimage import io
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from fastai import *
from fastai.vision import *
from fastai.layers import MSELossFlat, CrossEntropyFlat
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

class WildAnimals(Dataset):
    def __init__(self, transform=None, datasetLabel = "Train", size=64):
        self.bigHornCount = 0
        self.bobcatCount = 0
        self.coyoteCount = 0
        self.grayFoxCount = 0
        self.javelinaCount = 0
        self.muleDeerCount = 0
        self.raptorCount = 0
        self.whiteTailCount = 0
        self.dayCount = 0
        self.nightCount = 0
        self.total = 0
        self.datasetLabel = datasetLabel
        self.imgs_path = "resizedAnimals" + os.sep + datasetLabel + os.sep
        self.transform = transform
        file_list = glob.glob(self.imgs_path + "*")
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #imagenet stats
        self.size = size

        self.data = []
        for file_path in file_list:
            file = file_path.split(os.sep)[-1]
            #if any(x in file for x in chipMatches):
            self.updateStats(file)
            self.data.append([file_path, file])

        print(f"\n{datasetLabel}")
        #print("Chip:" + str(chipMatches))
        print("Bighorn_Sheep:" + str(self.bigHornCount))
        print("Bobcat:" +  str(self.bobcatCount))
        print("Coyote:" +  str(self.coyoteCount))
        print("Gray_Fox:" +  str(self.grayFoxCount))
        print("Javelina:" +  str(self.javelinaCount))
        print("Mule_Deer:" +  str(self.muleDeerCount))
        print("Raptor:" +  str(self.raptorCount))
        print("White_tailed_Deer:" +  str(self.whiteTailCount))
        print("Day:" +  str(self.dayCount))
        print("Night:" +  str(self.nightCount))
        print("Total:" +  str(self.total))

        self.saveDistrubution()

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

        self.day_map = {
        "day" : 0,
        "night": 1
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        img = PIL.Image.open(img_path).convert('RGB')
        img = Image(pil2tensor(img, dtype=np.float32).div_(255))
        img = img.apply_tfms(self.transform, size = self.size)
        img = self.norm(img.data)
        
        labels = label.split("-")
        # y_label = torch.tensor(self.class_map[labels[0]])
        # day_label = torch.tensor(self.day_map[labels[1]])

        type_label = torch.tensor(int(self.class_map[labels[0]]), dtype=torch.int64)
        day_label = torch.tensor(int(self.day_map[labels[1]]), dtype=torch.int64)
        
        return img.data, (type_label, day_label)
    
    def updateStats(self, img_name):
        if "Bighorn_Sheep" in img_name:
            self.bigHornCount += 1
        elif "Bobcat" in img_name:
            self.bobcatCount += 1
        elif "Coyote" in img_name:
            self.coyoteCount += 1
        elif "Gray_Fox" in img_name:
            self.grayFoxCount += 1
        elif "Javelina" in img_name:
            self.javelinaCount += 1
        elif "Mule_Deer" in img_name:
            self.muleDeerCount += 1
        elif "Raptor" in img_name:
            self.raptorCount += 1
        elif "White_tailed_Deer" in img_name:
            self.whiteTailCount += 1

        if "day" in img_name:
                self.dayCount += 1
        if "night" in img_name:
                self.nightCount += 1
        self.total += 1

    def saveDistrubution(self):
        dayPercent = (self.dayCount / self.total) * 100
        nightPercent = (self.nightCount / self.total) * 100
        y = np.array([dayPercent, nightPercent])
        labels = [f"Day {dayPercent:.2f}%", f"Night {nightPercent:.2f}%"]
        colors = ["yellow", "black"]

        plt.pie(y, labels = labels, colors=colors)
        plt.title(f"{self.datasetLabel} Time of Day Distribution, Total Images: {self.total}")
        plt.savefig(f'{self.datasetLabel}DataDistrubution.png')
        plt.close()
        
        #we could also add one for the animal distrubution

