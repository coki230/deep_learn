from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
print(x_train.shape)
print(len(x_train[0]))
print(len(x_train[1]))
a = pad_sequences(x_train, maxlen=500)
print(a.shape)
print(len(a[0]))