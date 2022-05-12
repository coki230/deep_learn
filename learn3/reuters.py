from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import optimizers
from keras import losses
from keras import layers
from keras import activations
import numpy as np

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000)

print(max(y_train))


def deal_data(data):
    ret = np.zeros(10000)
    ret[data] = 1
    return ret


def get_data(data):
    return np.array([deal_data(x) for x in data])


model = models.Sequential()
model.add(layers.Dense(512, activation=activations.relu, input_shape=(10000,)))
model.add(layers.Dense(128, activation=activations.relu))
model.add(layers.Dense(46, activation=activations.softmax))

model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.0001), loss=losses.categorical_crossentropy,
              metrics=["accuracy"])
model.fit(get_data(x_train), to_categorical(y_train), epochs=25, batch_size=128, validation_split=0.2)

ret = model.evaluate(get_data(x_test), to_categorical(y_test))
print(ret)
