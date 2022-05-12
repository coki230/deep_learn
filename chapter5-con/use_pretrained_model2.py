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


conv_base.trainable = False
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(loss=losses.binary_crossentropy, metrics=["acc"])
model.fit_generator(train_data,steps_per_epoch=10,epochs=10,validation_data=val_data,validation_steps=10)



