from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import keras.backend as bk
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

tf.compat.v1.disable_eager_execution()


img_path = "C:\\Users\\Coki_Zhao\\Pictures\\1.png"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x.reshape((1, 224, 224, 3))
x = preprocess_input(x)
model = VGG16()
preds = model.predict(x)
print(np.argmax(preds))
outputs = model.output[:, np.argmax(preds)]
last_layer = model.get_layer("block5_conv3")
print(decode_predictions(preds, top=3))
grads = bk.gradients(outputs, last_layer.output)[0]
pooled_greds = bk.mean(grads, axis=(0, 1, 2))
iterate = bk.function([model.input], [pooled_greds, last_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(heatmap)
# plt.imshow(heatmap)
# plt.show()

# img = cv2.imread(img_path)
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# cv2.imwrite("C:\\Users\\Coki_Zhao\\Pictures\\3.jpg", superimposed_img)

# plt.imshow(superimposed_img)
# plt.show()