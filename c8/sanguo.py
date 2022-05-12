import numpy as np
from keras import layers
from keras import models
from keras import activations
from keras import losses

file = open("san.txt", encoding="utf-8")
text = file.read()
maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i+maxlen])


chars = sorted(list(set(text)))
dic_char = dict((char, chars.index(char)) for char in chars)

train_data = np.zeros((len(sentences), maxlen, len(chars)))
train_label = np.zeros((len(sentences), len(chars)))

# fill the train data
for sentence_index in range(len(sentences)):
    for char_index, char_val in enumerate(sentences[sentence_index]):
        train_data[sentence_index, char_index, dic_char.get(char_val)] = 1
    train_label[sentence_index, dic_char.get(next_chars[sentence_index])] = 1
    # indexes = [dic_char.get(c) for c in sentences[sentence_index]]
    # train_data[sentence_index, indexes] = 1

# print(train_data.shape)
# print(train_label.shape)
# print(train_data[9])
# print(train_label[9])
# print(sentences[9])
# print(next_chars[9])

model = models.Sequential()
model.add(layers.LSTM(32, input_shape=(maxlen, len(chars),)))
model.add(layers.Dense(32, activation=activations.relu))
model.add(layers.Dense(len(chars), activation=activations.softmax))
model.compile(loss=losses.categorical_crossentropy)
model.fit(train_data, train_label, epochs=6, batch_size=20, validation_split=0.2)
model.save("san.h5")







