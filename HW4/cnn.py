#sources: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

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
import cv2
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from wildAnimalDataset import WildAnimals

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if os.path.exists("trainingIncorrect"):
    shutil.rmtree("trainingIncorrect")
ensure_dir("trainingIncorrect")

if os.path.exists("trainingCorrect"):
    shutil.rmtree("trainingCorrect")
ensure_dir("trainingCorrect")

if os.path.exists("testingIncorrect"):
    shutil.rmtree("testingIncorrect")
ensure_dir("testingIncorrect")

if os.path.exists("testingCorrect"):
    shutil.rmtree("testingCorrect")
ensure_dir("testingCorrect")

class CNN(nn.Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=numChannels, out_channels=20,
            kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize first (and only) set of FC => RELU layers
        # where does this number come from?
        self.fc1 = nn.Linear(in_features=186050, out_features=500)
        self.relu3 = nn.ReLU()

        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=500, out_features=classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output

if __name__ == '__main__': 
    transform = transforms.Compose(
        [transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Resize(256),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 16
    imageCount = 0

    datasetChip1 = WildAnimals(transform=transform)
    datasetChipOthers = WildAnimals(transform=transform, chipMatches=["chip02", "chip03", "chip04", "chip05", "chip06"])

    trainloader = torch.utils.data.DataLoader(datasetChip1, batch_size=batch_size, shuffle=True)

    testloader = torch.utils.data.DataLoader(datasetChipOthers, batch_size=batch_size, shuffle=True)

    classes = ('Bighorn_Sheep', 'Bobcat', 'Coyote', 'Gray_Fox',
            'Javelina', 'Mule_Deer', 'Raptor', 'White_tailed_Deer')

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
            inputs, labels, imgPath = data

            #print(type(inputs))
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = lossFn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)

            # collect the correct predictions for each class
            for label, prediction, imgPath in zip(labels, predictions, imgPath):
                time = imgPath.split("_")[-1].split(".")[-2]
                if label == prediction:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"trainingCorrect/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                    # cv2.imshow(f"trainingCorrect", image)
                    # cv2.waitKey(30)
                else:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"trainingIncorrect/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                    # cv2.imshow(f"trainingIncorrect", image)
                    # cv2.waitKey(30)

        # print statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(predictions == labels.data)

        # https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/
        # since micro averaging does not distinguish between different classes and then just averages their metric scores, this averaging scheme is not prone to inaccurate values due to an unequally distributed test set
        print(f'[{epoch + 1}] loss: {running_loss:3f} correct:{running_corrects}')
        print("F1 Score : ", f1_score(labels.cpu().data, predictions.cpu(), average='micro'))
        print("Precision Score : ", precision_score(labels.cpu().data, predictions.cpu(), average='micro'))
        print("Recall Score : ",recall_score(labels.cpu().data, predictions.cpu(), average='micro'))
        print("\n")

    print('Finished Training\n')

    dataiter = iter(testloader)
    images, labels, _ = dataiter.next()

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
                    # cv2.imshow(f"trainingCorrect", image)
                    # cv2.waitKey(30)
                else:
                    image = cv2.imread(imgPath)
                    cv2.imwrite(f"testingIncorrect/l_{classes[label]}_p_{classes[prediction]}_t_{time}_{imageCount}.jpg", image)
                    imageCount += 1
                    # cv2.imshow(f"trainingIncorrect", image)
                    # cv2.waitKey(30)
                total_pred[classes[label]] += 1

            #should we put this in a chart somehow? matplotlib.pyplot as plt
            print(f'[{batchId + 1}]')
            print("F1 Score : ", f1_score(labels.cpu().data, predictions.cpu(), average='micro'))
            print("Precision Score : ", precision_score(labels.cpu().data, predictions.cpu(), average='micro'))
            print("Recall Score : ",recall_score(labels.cpu().data, predictions.cpu(), average='micro'))
            print("\n")

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            accuracy = 0
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

#TODO read back in output data from predictions to look for trends
#TODO add scores to plots, epochs for x axis?: https://intellipaat.com/blog/tutorial/python-tutorial/python-matplotlib/