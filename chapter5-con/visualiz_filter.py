import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras import backend as bk
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

model = VGG16(weights="imagenet", include_top=False)
layer_name = "block3_conv1"
filter_index = 0


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = bk.mean(layer_output[:, :, :, filter_index])
    grads = bk.gradients(loss, model.input)[0]
    grads /= (bk.sqrt(bk.mean(bk.square(grads))) + 1e-5)
    iterate = bk.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, 150, 150, 3))
    print(input_img_data)
    step = 1
    for i in range(40):
        loss_val, grads_val = iterate([input_img_data])
        input_img_data += grads_val * step
    img = input_img_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern(layer_name, filter_index))
plt.show()
