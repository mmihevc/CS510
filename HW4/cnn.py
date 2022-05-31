#sources: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
#https://www.w3schools.com/python/matplotlib_pie_charts.asp
#https://github.com/aladdinpersson/Machine-Learning-Collection 

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

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if os.path.exists("trainingIncorrect"):
    shutil.rmtree("trainingIncorrect")
ensure_dir(os.path.join("trainingIncorrect", "epoch1"))
ensure_dir(os.path.join("trainingIncorrect", "epoch2"))
ensure_dir(os.path.join("trainingIncorrect", "epoch3"))
ensure_dir(os.path.join("trainingIncorrect", "epoch4"))
ensure_dir(os.path.join("trainingIncorrect", "epoch5"))

if os.path.exists("trainingCorrect"):
    shutil.rmtree("trainingCorrect")
ensure_dir(os.path.join("trainingCorrect", "epoch1"))
ensure_dir(os.path.join("trainingCorrect", "epoch2"))
ensure_dir(os.path.join("trainingCorrect", "epoch3"))
ensure_dir(os.path.join("trainingCorrect", "epoch4"))
ensure_dir(os.path.join("trainingCorrect", "epoch5"))

if os.path.exists("testingIncorrect"):
    shutil.rmtree("testingIncorrect")
ensure_dir("testingIncorrect")

if os.path.exists("testingCorrect"):
    shutil.rmtree("testingCorrect")
ensure_dir("testingCorrect")


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, type, day):

        mse, crossEntropy = MSELossFlat(), CrossEntropyFlat()

        loss0 = mse(preds[0], type)
        loss1 = crossEntropy(preds[1],day)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]
        
        return loss0+loss1

# model code came from this really helpful tutorial: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
# We also made some tweaks to work correctly with our images namely the in features
# class CNN(nn.Module):
#     def __init__(self, numChannels, classes):
#         # call the parent constructor
#         super(CNN, self).__init__()
#         self.flatten = nn.Flatten()

#         # initialize first set of CONV => RELU => POOL layers
#         self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
#             kernel_size=(5, 5))
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

#         # initialize second set of CONV => RELU => POOL layers
#         self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
#             kernel_size=(5, 5))
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

#         # initialize first (and only) set of FC => RELU layers
#         # where does this number come from?
#         self.fc1 = nn.Linear(in_features=186050, out_features=500)
#         self.relu3 = nn.ReLU()

#         # initialize our softmax classifier
#         self.fc2 = nn.Linear(in_features=500, out_features=classes)
#         self.logSoftmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         # pass the input through our first set of CONV => RELU =>
#         # POOL layers
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)

#         # pass the output from the previous layer through the second
#         # set of CONV => RELU => POOL layers
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)

#         # flatten the output from the previous layer and pass it
#         # through our only set of FC => RELU layers
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu3(x)

#         # pass the output to our softmax classifier to get our output
#         # predictions
#         x = self.fc2(x)
#         output = self.logSoftmax(x)
#         # return the output predictions
#         return output

class MultiTaskModel(nn.Module):
    """
    Creates a MTL model with the encoder from "arch" and with dropout multiplier ps.
    """
    def __init__(self, arch,ps=0.5):
        super(MultiTaskModel,self).__init__()
        self.encoder = create_body(arch)        #fastai function that creates an encoder given an architecture
        self.fc1 = create_head(1024,1,ps=ps)    #fastai function that creates a head
        self.fc2 = create_head(1024,2,ps=ps)
        #self.fc3 = create_head(1024,5,ps=ps)

    def forward(self,x):

        x = self.encoder(x)
        age = torch.sigmoid(self.fc1(x))
        gender = self.fc2(x)
        ethnicity = self.fc3(x)

        return [age, gender, ethnicity]

if __name__ == '__main__': 
    transform = transforms.Compose(
        [transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16
    imageCount = 0

    datasetChip1 = WildAnimals(transform=transform)
    datasetChipOthers = WildAnimals(transform=transform, chipMatches=["chip02", "chip03", "chip04", "chip05", "chip06"], datasetLabel="Test")

    trainloader = torch.utils.data.DataLoader(datasetChip1, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(datasetChipOthers, batch_size=batch_size, shuffle=True)

    classes = ('Bighorn_Sheep', 'Bobcat', 'Coyote', 'Gray_Fox',
            'Javelina', 'Mule_Deer', 'Raptor', 'White_tailed_Deer')

    timesOfDay = ('day', 'night')

    net = CNN(3, 8)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    lossFn = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0
        running_corrects = 0.0

        for batchId, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, dayLabels, imgPath = data

            #print(type(inputs))
            inputs = inputs.to(device)
            labels = labels.to(device)
            dayLabels = dayLabels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = lossFn(outputs, labels, dayLabels)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction, imgPath in zip(labels, predictions, imgPath):
                time = imgPath.split("_")[-1].split(".")[-2]
                if label == prediction:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"trainingCorrect/epoch{epoch + 1}/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                    # cv2.imshow(f"trainingCorrect", image)
                    # cv2.waitKey(30)
                else:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"trainingIncorrect/epoch{epoch + 1}/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                    # cv2.imshow(f"trainingIncorrect", image)
                    # cv2.waitKey(30)

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)

        # https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/
        # since micro averaging does not distinguish between different classes and then just averages their metric scores, this averaging scheme is not prone to inaccurate values due to an unequally distributed test set
        print(f'\n[{epoch + 1}] loss: {running_loss:3f} correct:{running_corrects}')
        print("F1 Score : ", f1_score(labels.cpu().data, predictions.cpu(), average='micro'))
        print("Precision Score : ", precision_score(labels.cpu().data, predictions.cpu(), average='micro'))
        print("Recall Score : ",recall_score(labels.cpu().data, predictions.cpu(), average='micro'))

    print('Finished Training\n')

    dataiter = iter(testloader)
    images, labels, _ = dataiter.next()
    fone_scores = []

    # again no gradients needed
    with torch.no_grad():
        #for epoch in range(5): epochs for test data?

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        for batchId, data in enumerate(testloader, 0):
            images, labels, imgPath = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to(device)
            # collect the correct predictions for each class
            for label, prediction, imgPath in zip(labels, predictions, imgPath):
                time = imgPath.split("_")[-1].split(".")[-2]
                if label == prediction:
                    correct_pred[classes[label]] += 1
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"testingCorrect/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                else:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"testingIncorrect/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                total_pred[classes[label]] += 1

            #should we put this in a chart somehow? matplotlib.pyplot as plt
            f1Score = f1_score(labels.cpu().data, predictions.cpu(), average='micro')
            fone_scores.append(f1Score)
            print(f'[{batchId + 1}]')
            print("F1 Score : ", f1Score)
            print("Precision Score : ", precision_score(labels.cpu().data, predictions.cpu(), average='micro'))
            print("Recall Score : ",recall_score(labels.cpu().data, predictions.cpu(), average='micro'))
            print("\n")

    # Build F1 Score line graph
    batch_x = np.arange(1, 14)
    f1 = np.array(fone_scores)
    plt.title("Testing F1 Scores")
    plt.xlabel("Batch ID")
    plt.ylabel("F1 Score")
    plt.plot(batch_x, f1)
    plt.savefig(f'testingF1Scores.png')
    plt.close()

    # read back in output data from predictions to look for trends
    # https://intellipaat.com/blog/tutorial/python-tutorial/python-matplotlib/

    #read back in the outputs to create charts
    def returnDatNightStats (imgs_path):
        dayCount = 0
        nightCount = 0
        total = 0
        file_list = glob.glob(imgs_path + "*")
        for file_path in file_list:
            file = file_path.split(os.sep)[-1]
            if "day" in file:
                    dayCount += 1
            if "night" in file:
                    nightCount += 1
            total += 1
        return dayCount, nightCount, total

    dayCountIn, nightCountIn, totalIn = returnDatNightStats("testingIncorrect" + os.sep)
    dayCountCor, nightCountCor, totalCor = returnDatNightStats("testingCorrect" + os.sep)

    total = totalIn + totalCor
    dayPercentIn = (dayCountIn / total) * 100
    nightPercentIn = (nightCountIn / total) * 100
    dayPercentCor = (dayCountCor / total) * 100
    nightPercentCor = (nightCountCor / total) * 100
    y = np.array([dayPercentCor, dayPercentIn, nightPercentCor, nightPercentIn])
    labels = [f"Day Correct\n{dayPercentCor:.2f}%",
    f"Day Incorrect\n{dayPercentIn:.2f}%",
    f"Night Correct\n{nightPercentCor:.2f}%",
    f"Night Incorrect\n{nightPercentIn:.2f}%"]

    plt.pie(y, labels = labels)
    plt.title(f"Testing Predictions, Total Images: {total}")
    plt.savefig('testPredictions.png')
    plt.close()

    for epoch in range(5):
        
        dayCountIn, nightCountIn, totalIn = returnDatNightStats(os.path.join("trainingIncorrect", f"epoch{epoch + 1}") + os.sep)
        dayCountCor, nightCountCor, totalCor = returnDatNightStats(os.path.join("trainingCorrect", f"epoch{epoch + 1}") + os.sep)

        total = totalIn + totalCor
        dayPercentIn = (dayCountIn / total) * 100
        nightPercentIn = (nightCountIn / total) * 100
        dayPercentCor = (dayCountCor / total) * 100
        nightPercentCor = (nightCountCor / total) * 100
        y = np.array([dayPercentCor, dayPercentIn, nightPercentCor, nightPercentIn])
        labels = [f"Day Correct\n{dayPercentCor:.2f}%",
        f"Day Incorrect\n{dayPercentIn:.2f}%",
        f"Night Correct\n{nightPercentCor:.2f}%",
        f"Night Incorrect\n{nightPercentIn:.2f}%"]

        plt.pie(y, labels = labels)
        plt.title(f"Training Predictions Epoch {epoch + 1}, Total Images: {total}")
        plt.savefig(f'trainingPredictionsEpoch{epoch + 1}.png')
        plt.close()

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')