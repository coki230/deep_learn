from keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras import activations
from keras import losses
from keras import optimizers


train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

train_dir = "C:\\Users\\Coki_Zhao\\Desktop\\data\\archive\\training_set\\training_set"
train_gen = train_data.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode="binary")

val_dir = "C:\\Users\\Coki_Zhao\\Desktop\\data\\archive\\training_set\\validation_set"
val_gen = train_data.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20, class_mode="binary")

# for data, label in train_gen:
#     print("data shape: ", data.shape)
#     for i in range(data.shape[0]):
#         plt.imshow(data[i])
#         plt.show()
#     print("label shape: ", label.shape)
#     print(label)
#     break

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation=activations.relu, input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation=activations.relu))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=activations.relu))
model.add(layers.Dense(1, activation=activations.sigmoid))

model.compile(loss=losses.binary_crossentropy, metrics=["acc"])
model.fit_generator(train_gen, steps_per_epoch=100, epochs=20, validation_data=val_gen, validation_steps=20)

model.save("cat_dag.h5")
