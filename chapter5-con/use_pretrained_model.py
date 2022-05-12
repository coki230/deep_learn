from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras import activations
from keras import losses
from keras.utils.np_utils import to_categorical

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))

train_dir = "C:\\Users\\Coki_Zhao\\Desktop\\data\\archive\\training_set\\training_set"
val_dir = "C:\\Users\\Coki_Zhao\\Desktop\\data\\archive\\training_set\\validation_set"
train_gen = ImageDataGenerator(rescale=1. / 255)
train_data = train_gen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=200, class_mode="binary")
val_gen = ImageDataGenerator(rescale=1. / 255)
val_data = val_gen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=200, class_mode="binary")


train_use_data = np.empty(0)
train_use_label = np.empty(0)
for data, label in train_data:
    ret = conv_base.predict(data)
    if train_use_data.shape == (0,):
        train_use_data = ret
    else:
        train_use_data = np.append(ret, train_use_data, axis=0)
    if train_use_label.shape == (0,):
        train_use_label = label
    else:
        train_use_label = np.append(label, train_use_label, axis=0)

    print(train_use_label.shape)
    print(train_use_data.shape)
    print("=======================")
    if train_use_data.shape[0] > 1000:
        break

train_use_data = np.reshape(train_use_data, (train_use_data.shape[0], -1))
print(train_use_label.shape)
print(train_use_data.shape)

model = models.Sequential()
model.add(layers.Dense(64, activation=activations.relu, input_shape=(4*4*512,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(loss=losses.binary_crossentropy, metrics=["acc"])
model.fit(train_use_data, train_use_label, batch_size=20, epochs=10, validation_split=0.2)



