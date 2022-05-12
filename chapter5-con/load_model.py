from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras import models

model = load_model("cat_dag.h5")
# model.summary()

img_path = "C:\\Users\\Coki_Zhao\\Desktop\\data\\archive\\training_set\\training_set\\cats\\cat.401.jpg"
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
print(img_tensor.shape)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
#
# plt.imshow(img_tensor[0])
# plt.show()
out_layers = [layer.output for layer in model.layers[:8]]
act_model = models.Model(inputs=model.input, outputs=out_layers)
activations = act_model.predict(img_tensor)
first_layer = activations[0]
print(first_layer.shape)
for lay in activations:
    print(lay.shape)

big_img = np.zeros((148 * 8, 148 * 8))
for col in range(8):
    for row in range(8):
        data = first_layer[0, :, :, row + col * 8]
        data = data.reshape(148, 148)
        big_img[col * 148: (col + 1) * 148, row * 148: (row + 1) * 148] = data

plt.figure(figsize=(20, 20))
plt.imshow(big_img)
plt.show()
