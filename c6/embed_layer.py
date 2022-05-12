from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras import activations
from keras import losses

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
x_train = pad_sequences(x_train, maxlen=100)
x_test = pad_sequences(x_test, maxlen=100)

model = models.Sequential()
model.add(layers.Embedding(10000, 8, input_length=100))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation=activations.sigmoid))
model.compile(loss=losses.binary_crossentropy, metrics=["acc"])
model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
