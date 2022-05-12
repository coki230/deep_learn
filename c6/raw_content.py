import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

data_path = "C:\\Users\\Coki_Zhao\\Desktop\\data\\aclImdb\\aclImdb\\test"
g = os.walk(data_path)
raw_text = []
labels = []
for a, b, c in g:
    if len(b) != 0:
        continue
    for son in c:
        file = open(a + "\\" + son, encoding="utf8")
        raw_text.append(file.read())
        if a == "neg":
            labels.append(0)
        else:
            labels.append(1)

tokenizer = Tokenizer(1000)
tokenizer.fit_on_texts(raw_text)
seq = tokenizer.texts_to_sequences(raw_text)
data = pad_sequences(seq, maxlen=100)
num = data.shape[0]
indexes = range(num)
np.random.shuffle(indexes)
data = data[indexes]
labels = labels[indexes]


