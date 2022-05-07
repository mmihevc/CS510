from logging.handlers import BaseRotatingHandler
from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import glob
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from wildAnimalDataset import WildAnimals
from fastai import *
from fastai.vision import *
from fastai.layers import MSELossFlat, CrossEntropyFlat
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

# things to write about in our paper
# multi task training
# FastAI
# dataset and results


# Followed this example: https://towardsdatascience.com/multi-task-learning-with-pytorch-and-fastai-6d10dc7ce855
# https://iconof.com/1cycle-learning-rate-policy/ 
class MultiTaskModel(nn.Module):
    def __init__(self, arch,ps=0.5):
        super(MultiTaskModel,self).__init__()
        self.encoder = create_body(arch)
        self.fc1 = create_head(1024,8,ps=ps) # number of classes as out.
        self.fc2 = create_head(1024,2,ps=ps)

    def forward(self,x):
        x = self.encoder(x)
        classType = torch.sigmoid(self.fc1(x))
        time = self.fc2(x)

        return [classType, time]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, classType, time):

        mse, crossEntropy = MSELossFlat(), CrossEntropyFlat()

        loss0 = crossEntropy(preds[0],classType)
        loss1 = crossEntropy(preds[1],time)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        # precision2 = torch.exp(-self.log_vars[2])
        # loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1

class Predictor():
    def __init__(self, model):
        self.model = model
        self.tfms = get_transforms()[1]
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #imagenet stats
        self.timeOfDay = {0:"day",1:"night"}
        self.typeOfAnimal = {0:"Bighorn_Sheep",1:"Bobcat",2:"Coyote",3:"Gray_Fox",4:"Javelina",
        5:"Mule_Deer", 6:"Raptor", 7:"White_tailed_Deer"}

    def predict(self,x):
        #x is a PIL Image
        x = Image(pil2tensor(x, dtype=np.float32).div_(255))
        x = x.apply_tfms(self.tfms, size = 64)
        x = self.norm(x.data)
        preds = self.model(x.unsqueeze(0))
        classType = self.typeOfAnimal[torch.softmax(preds[0],1).argmax().item()]
        time = self.timeOfDay[torch.softmax(preds[1],1).argmax().item()]
        return classType, time

if __name__ == '__main__': 
    batch_size = 16
    imageCount = 0
    tfms = get_transforms()

    datasetChip1 = WildAnimals(transform=tfms[0])
    datasetChipOthers = WildAnimals(transform=tfms[1], datasetLabel="Test")

    trainloader = torch.utils.data.DataLoader(datasetChip1, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(datasetChipOthers, batch_size=batch_size, shuffle=True)
    data = DataBunch(train_dl=trainloader, valid_dl=trainloader, test_dl=testloader)

    def acc_class_type(preds, classType, time): return accuracy(preds[0], classType)
    def acc_time(preds, classType, time): return accuracy(preds[1], time)
    metrics = [acc_class_type, acc_time]

    model = MultiTaskModel(models.resnet34, ps=0.25)
    #model.eval()

    loss_func = MultiTaskLossWrapper(2).to(data.device) #just making sure the loss is on the gpu

    learn = Learner(data, model, loss_func=loss_func, callback_fns=ShowGraph, metrics=metrics)

    #spliting the model so that I can use discriminative learning rates
    learn.split([learn.model.encoder[:6],
                learn.model.encoder[6:],
                nn.ModuleList([learn.model.fc1, learn.model.fc2])])

    #first I'll train only the last layer group (the heads)
    learn.freeze()

    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties(0))

    learn.fit_one_cycle(5,max_lr=1e-2,
                    callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='stage-1')])

    learn.load("stage-1")
    learn.unfreeze()

    def PredictTestSet(pred):
        correctAnimal = 0
        incorrectAnimal = 0
        correctTime = 0
        incorrectTime = 0
        bothCorrect = 0
        oneIncorrect = 0
        inputFolder = "resizedAnimals\\Test"
        paths = glob.glob(inputFolder + "*")
        for class_path in paths:
            for img_path in glob.glob(class_path + "/*.jpg"):
                try:
                    img = PIL.Image.open(img_path)
                    predAnimal, predTime = pred.predict(img)

                    labels = img_path.split("\\")[-1].split("-")
                    actualAnimal = labels[0]
                    actualTime = labels[1]

                    if(predAnimal == actualAnimal):
                        correctAnimal += 1
                    else:
                        incorrectAnimal += 1

                    if(predTime == actualTime):
                        correctTime += 1
                    else:
                        incorrectTime += 1

                    if(predAnimal == actualAnimal and predTime == actualTime):
                        bothCorrect += 1
                    else:
                        oneIncorrect += 1
                except:
                    print("An exception occurred")

        return correctAnimal, incorrectAnimal, correctTime, incorrectTime, bothCorrect, oneIncorrect

    trained_model = learn.model.cpu()
    pred = Predictor(trained_model)
    correctAnimal, incorrectAnimal, correctTime, incorrectTime, bothCorrect, oneIncorrect = PredictTestSet(pred)

    total = correctAnimal + incorrectAnimal
    animalPercentIn = (incorrectAnimal / total) * 100
    animalPercentCor = (correctAnimal / total) * 100

    y = np.array([animalPercentCor, animalPercentIn])
    labels = [f"Animal Correct\n{animalPercentCor:.2f}%",
    f"Animal Incorrect\n{animalPercentIn:.2f}%"]

    plt.pie(y, labels = labels)
    plt.title(f"Testing Predictions Animals, Total Images: {total}")
    plt.savefig('testPredictionsAnimals.png')
    plt.close()

    total = bothCorrect + oneIncorrect
    bothPercentIn = (oneIncorrect / total) * 100
    bothPercentCor = (bothCorrect / total) * 100

    y = np.array([bothPercentCor, bothPercentIn])
    labels = [f"Both Correct\n{bothPercentCor:.2f}%",
    f"At least one Incorrect\n{bothPercentIn:.2f}%"]

    plt.pie(y, labels = labels)
    plt.title(f"Testing Predictions Both Classes, Total Images: {total}")
    plt.savefig('testPredictionsBothClasses.png')
    plt.close()

    total = correctTime + incorrectTime
    timePercentIn = (incorrectTime / total) * 100
    timePercentCor = (correctTime / total) * 100

    y = np.array([timePercentCor, timePercentIn])
    labels = [f"Time Correct\n{timePercentCor:.2f}%",
    f"Time Incorrect\n{timePercentIn:.2f}%"]

    plt.pie(y, labels = labels)
    plt.title(f"Testing Predictions Time of Day, Total Images: {total}")
    plt.savefig('testPredictionsTimeOfDay.png')
    plt.close()
