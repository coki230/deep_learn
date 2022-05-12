from keras.datasets import boston_housing
from keras.models import Sequential
from keras import activations
from keras import layers
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt
from keras import regularizers

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()


# print(x_train.shape)
# print(y_train.shape)
#
# print(x_train[0])
# print(y_train[0])

def normalize_data(data):
    data = data - data.mean(axis=0)
    data = data / data.std(axis=0)
    return data


def build_model():
    model = Sequential()
    model.add(layers.Dense(64, activation=activations.relu, input_shape=(13,)))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(1))
    return model

def build_model2():
    model = Sequential()
    model.add(layers.Dense(64, activation=activations.relu, kernel_regularizer=regularizers.l2(), input_shape=(13,)))
    model.add(layers.Dense(64, activation=activations.relu, kernel_regularizer=regularizers.l2()))
    model.add(layers.Dense(1))
    return model


model = build_model()
model.compile(optimizer=optimizers.adam_v2.Adam(), loss=losses.mean_absolute_error)
his = model.fit(normalize_data(x_train), y_train, batch_size=1, epochs=30, validation_split=0.2)


model2 = build_model2()
model2.compile(optimizer=optimizers.adam_v2.Adam(), loss=losses.mean_absolute_error)
his2 = model2.fit(normalize_data(x_train), y_train, batch_size=1, epochs=30, validation_split=0.2)

plt.plot(his.epoch[3:], his.history["loss"][3:], "bo")
plt.plot(his.epoch[3:], his.history["val_loss"][3:], "rx")
plt.plot(his2.epoch[3:], his2.history["loss"][3:], "y<")
plt.plot(his2.epoch[3:], his2.history["val_loss"][3:], "g>")
plt.show()

