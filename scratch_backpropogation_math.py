import numpy as np

weight = 0.7
bias = 0.3

X = np.arange(0, 1, 0.02).reshape(-1, 2)
y = weight * X.sum(axis=1, keepdims=True) + bias

split = int(0.8 * len(X))

X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

np.random.seed(42)
lr = 0.01

layer_weight = np.random.randn(2, 2)
layer_bias = np.random.randn(2)


class Layer:
    def __init__(self, n_input, n_neurons):
        self.weights = np.random.randn(n_input, n_neurons)
        self.bias = np.random.randn(n_neurons)

    def forward(self, X):
        self.last_input = X
        return np.dot(X, self.weights) + self.bias


network = [
    Layer(2, 64),
    Layer(64, 64),
    Layer(64, 64),
    Layer(64, 1),
]

output = X_train
for layer in network:
    output = layer.forward(ouput)

error = output - y_train
for layer in reversed(network):
    error = layer.backward(error)

for layer in network:
    layer.weights -= lr * layer.grad_w
    layer.bias -= lr * layer.grad_b

# def forward(X, weight, bias):  # this is the model.train()
#     return np.dot(X, weight) + bias
#
#
# def loss_fn(y_pred):
#     loss_fn = np.mean(np.abs(y_pred - y_train))
#     return loss_fn
#
#
# def backward(y_pred, y_true, X):
#     error = y_pred - y_true
#     grad_w = np.dot(X.T, error) / len(y_true)
#     grad_b = np.mean(error)
#     return grad_w, grad_b
#
#
# def grad_zero(grad_shape, bias_shape):
#     grad_w = np.zeros(grad_shape)
#     grad_b = np.zeros(bias_shape)
#     return grad_w, grad_b
#
#
# def optimizer(gradients, old):
#     new_weights = old - (lr * gradients)
#     return new_weights
#
#
# epochs = 500
# for epoch in range(epochs):
#     grad_w, grad_b = grad_zero(layer_weight.shape, layer_bias.shape)
#     y_pred = forward(X_train, layer_weight, layer_bias)
#     loss = loss_fn(y_pred)
#     grad_w, grad_b = backward(y_pred, y_train, X_train)
#     layer_weight = optimizer(grad_w, layer_weight)
#     layer_bias = optimizer(grad_b, layer_bias)
#
#     if epoch % 10 == 0:
#         print(loss)
