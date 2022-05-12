import keras.callbacks
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras import losses
from keras import activations
import numpy as np

max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train.shape)
x_train = pad_sequences(x_train, maxlen=max_len)

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name="embed"))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation=activations.relu))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()
model.compile(loss=losses.binary_crossentropy, metrics=["acc"])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir="log_dir",
        histogram_freq=1,
        embeddings_freq=1
    )
]

his = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)