from PIL import Image
import numpy as np


def convert(file):
    im = Image.open(file)
    im = im.convert("L")
    im = im.resize((28, 28))
    vector = np.asarray(im)
    vector = vector.reshape(1, -1)
    return vector

