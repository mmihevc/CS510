import tensorflow as tf

## soruce from https://github.com/Iqbal1282/Vgg16/blob/master/vgg16_model.py

input_layer = tf.keras.layers.Input([128, 128, 3])
# block1
conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv1_1')(input_layer)

conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv1_2')(conv1_1)

pool1_1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1_1')

# block 2
conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv2_1')(pool1_1)

conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv2_2')(conv2_1)

pool2_1 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2_1')

# block3

conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv3_1')(pool2_1)

conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv3_2')(conv3_1)

conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv3_3')(conv3_2)

conv3_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv3_4')(conv3_3)

pool3_1 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding="SAME", name='pool3_1')

# block4

conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv4_1')(pool3_1)

conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv4_2')(conv4_1)

conv4_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv4_3')(conv4_2)

conv4_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv4_4')(conv4_3)

pool4_1 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool4_1')

# block5


conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv5_1')(pool4_1)

conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv5_2')(conv5_1)

conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv5_3')(conv5_2)

conv5_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=[1, 1],
                                 padding='same', activation='relu', name='conv5_4')(conv5_3)

pool5_1 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool5_1')

flatten = tf.keras.layers.Flatten()(pool5_1)

fc6 = tf.keras.layers.Dense(units=512, name='fc6', activation='relu')(flatten)
fc7 = tf.keras.layers.Dense(units=512, name='fc7', activation='relu')(fc6)
fc8 = tf.keras.layers.Dense(units=8, name='fc8', activation=None)(fc7)

prob = tf.nn.softmax(fc8)

model = tf.keras.Model(input_layer, prob)
model.summary()

model.save('myVgg16.h5')
