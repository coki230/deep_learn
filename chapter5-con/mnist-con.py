from keras.datasets import mnist
from keras import models
from keras import losses
from keras import layers
from keras import activations

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_test.shape)
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation=activations.relu, input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(63, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(63, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(10, activation=activations.softmax))

model.compile(loss=losses.sparse_categorical_crossentropy, metrics=["acc"])
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)



