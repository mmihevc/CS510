from DataOrg import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
classes = ["Bighorn_Sheep", "Bobcat" ,"Coyote" ,"Gray_Fox","Javelina", "Mule_Deer", "Raptor","White_tailed_Deer"]
file = "dataChips.pickle"
def main():
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(8))

    features, label = load_data(file)
    plt.figure(figsize=(8, 8))
    for i in range(25):
        plt.subplot(6, 5, i + 1)
        plt.imshow(features[i])
        plt.xlabel(classes[label[i]])
        plt.xticks([])
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, shuffle=True,random_state=1001)
    print(type(x_train))
    model = tf.keras.models.load_model(f'myVgg16.h5')
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.2, epochs=5)
    model.save(f'Models/VGGTrained0-5.h5')
    print(history)
    for i in range(1,5):
        model = tf.keras.models.load_model(f'Models/VGGTrained{(i-1)*5}-{i*5}.h5')
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=5)
        model.save(f'Models/VGGTrained{(i)*5}-{i*5+5}.h5')


if __name__ == '__main__':
    main()