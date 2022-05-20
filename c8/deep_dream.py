from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

model = InceptionV3(include_top=False)
layer_contributions = {
    "mixed2": 0.2,
    "mixed3": 3,
    "mixed4": 2,
    "mixed5": 1.5
}
layer_dict = dict([(layer.name, layer) for layer in model.layers])
loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), "float32"))
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2:-2, :]))

dream = model.input
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print("loss value at ", i, ": ", loss_value)
        x += step * grad_values
    return x
