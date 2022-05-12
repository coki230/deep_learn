from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras import layers
from keras import activations
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
dic = imdb.get_word_index()


reversed_dic = dict([(val, key) for (key, val) in dic.items()])

def fill_text(x):
    r = np.zeros(10000)
    r[x] = 1
    return r

deal_train = np.array([fill_text(x) for x in x_train])
deal_test = np.array([fill_text(x) for x in x_test])


model = Sequential()
model.add(layers.Dense(16, activation=activations.relu, input_shape=(10000,)))
model.add(layers.Dense(16, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.0001), loss=losses.mean_absolute_error, metrics=["accuracy"])
his = model.fit(deal_train, y_train, batch_size=128, epochs=12, validation_split=0.2)

test = model.evaluate(deal_test, y_test)

print(test)
# plt.plot(his.epoch, his.history["loss"], color="red")
# plt.scatter(x=his.epoch, y=his.history["val_loss"], color="blue")
# plt.show()
