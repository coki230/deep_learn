import numpy as np
from keras import models
from keras import activations
from keras import losses
from keras import layers
from keras import optimizers

data_path = "C:\\Users\\Coki_Zhao\\Desktop\\data\\jena_climate_2009_2016.csv"
f = open(data_path)
data = f.read()
f.close()
line_data = data.split("\n")
head = line_data[0]
train_data = line_data[1:20000]
# how many data will look
lookback = 720
# sample data period
step = 6
# timestep for predict data
delay = 144


def get_data_label(d_data):
    len_num = len(d_data)
    labels = []
    data_process = []
    ret_data = []
    for index in range(len_num):
        one_data = get_data_detail(d_data[index])
        data_process.append(one_data)
    data_process = np.array(data_process)
    data_process -= data_process.mean(axis=0)
    std = data_process.std(axis=0)
    data_process /= std
    print("============", std)
    for i in range(lookback, len_num, delay):
        indexes = range(i - 720, i, step)
        ret_data.append(data_process[indexes])
        predict_index = i + 144
        if predict_index >= len_num:
            predict_index = i
        try:
            labels.append(data_process[predict_index][1])
        except IndexError:
            print(len_num)
            print(predict_index)

    return np.array(ret_data, dtype=np.float_), np.array(labels, dtype=np.float_)


def get_data_detail(d_data):
    ret = d_data.split(",")
    val = [float(v) for v in ret[1:]]
    return val


def get_base_line(d_data):
    data_num = len(d_data)
    def_data = []
    for i in range(data_num):
        if i < data_num - 1:
            now_data = d_data[i][0][1]
            after_data = d_data[i + 1][0][1]
            def_data.append(float(after_data) - float(now_data))
    def_data = np.absolute(def_data)
    return np.mean(def_data)


# print(head)
# print(len(head.split(",")))
data, label = get_data_label(train_data)
print(data.shape)
print(len(data))
print(get_base_line(data))
# base 0.3182561489565783
model = models.Sequential()
# no 1 loss: 0.2965 - val_loss: 0.2494
# model.add(layers.GRU(32, input_shape=(120, 14), return_sequences=True, recurrent_dropout=0.5, dropout=0.1))
# model.add(layers.GRU(32, recurrent_dropout=0.5, dropout=0.1))

# no 2 loss: 0.2755 - val_loss: 0.3090
# model.add(layers.LSTM(32, input_shape=(120, 14), recurrent_dropout=0.5, dropout=0.1))

# no 3 loss: 0.4012 - val_loss: 1.2624
# model.add(layers.Flatten(input_shape=(120, 14)))
# model.add(layers.Dense(32, activation=activations.relu))

# no 4 loss: 0.2104 - val_loss: 0.4243
model.add(layers.Flatten(input_shape=(120, 14)))
model.add(layers.Dense(32, activation=activations.relu))

model.add(layers.Dense(1))
model.compile(loss=losses.mae)
model.summary()
model.fit(data, label, epochs=20, batch_size=500, validation_split=0.2)
