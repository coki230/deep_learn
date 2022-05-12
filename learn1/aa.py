from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import activations
from keras import losses
from keras import metrics
import deal_image as di

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
print(x_train.shape)
print(x_train.shape[0])


net = models.Sequential()
net.add(layers.Dense(32, activation=activations.relu, input_shape=(x_train.shape[1],)))
net.add(layers.Dense(32, activation=activations.relu))
net.add(layers.Dense(10, activation=activations.softmax))

net.compile(loss=losses.sparse_categorical_crossentropy, metrics=["accuracy"])
net.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.8)

# ret = net.predict(di.convert("img/6.jpg"))
#
#
# print(max(ret[0]))
# print(ret[0])