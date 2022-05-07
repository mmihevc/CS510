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

if __name__ == '__main__': 
    batch_size = 16
    imageCount = 0
    tfms = get_transforms()

    datasetChip1 = WildAnimals(transform=tfms[0])
    datasetChipOthers = WildAnimals(transform=tfms[1], chipMatches=["chip02", "chip03", "chip04", "chip05", "chip06"], datasetLabel="Test")

    trainloader = torch.utils.data.DataLoader(datasetChip1, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(datasetChipOthers, batch_size=batch_size, shuffle=True)
    data = DataBunch(trainloader, testloader)


    classes = ('Bighorn_Sheep', 'Bobcat', 'Coyote', 'Gray_Fox',
            'Javelina', 'Mule_Deer', 'Raptor', 'White_tailed_Deer')

    timesOfDay = ('day', 'night')

    def acc_class_type(preds, classType, time): return accuracy(preds[0], classType)
    def acc_time(preds, classType, time): return accuracy(preds[1], time)
    metrics = [acc_class_type, acc_time]

    model = MultiTaskModel(models.resnet34, ps=0.25)

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

    learn.lr_find()
    learn.recorder.plot()
    plt.savefig('test.png')

    learn.fit_one_cycle(15,max_lr=1e-2,
                    callbacks=[callbacks.SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='stage-1')])

    learn.load("stage-1")
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot()
    plt.savefig('test2.png')

