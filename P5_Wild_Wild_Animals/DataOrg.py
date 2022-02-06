import os
import pickle
import cv2
import numpy as np
import tensorflow_hub as hub
classes = ["Bighorn_Sheep", "Bobcat" ,"Coyote" ,"Gray_Fox","Javelina", "Mule_Deer", "Raptor","White_tailed_Deer"]
path = "More_data"
MAX_SIZE=(128,128)
def format_data():
    data = []
    count = 0
    directory = os.listdir(path)
    for file in directory:
        if count % 20 == 0:
            print(f"On count: {count}")
        if os.path.isdir(f"{path}/{file}"):
            for f in os.listdir(f"{path}/{file}"):
                data.append(get_info(f"{path}/{file}", f))
        count += 1
        data.append(get_info(path,file))
    data = [i for i in data if i]
    return data

def get_info(p,file):
    img = cv2.imread(f"{p}/{file}")
    if img is not None:
        imCrop = cv2.resize(img, MAX_SIZE)
        return [cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB),get_label(file)]

def get_label(file):
    for c in classes:
        if c in file:
            return classes.index(c)

def make_data():
    data=format_data()
    pik = open("dataMore128.pickle", 'wb')
    pickle.dump(data, pik)
    pik.close()

def load_data(f):
    pick = open(f, 'rb')
    data = pickle.load(pick)
    pick.close()
    features = []
    labels = []
    for img, label in data:
        features.append(np.array(img))
        labels.append(label)

    features = np.asarray(features).astype(np.float32)
    feature = features / 255.
    labels = np.array(labels)
    return feature,labels

def main():
    make_data()


if __name__ == '__main__':
    main()