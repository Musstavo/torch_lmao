import math
import numpy as np
from numpy.random import exponential

np.random.seed(42)


def read_binary_labels(file):
    raw_data = np.fromfile(file, dtype=np.uint8)
    raw_data = raw_data[8:]
    return raw_data


def read_binary_images(file):
    raw_data = np.fromfile(file, dtype=np.uint8)
    raw_data = raw_data[16:]
    return raw_data


train_labels = read_binary_labels("digits_data/train_labels")
train_images = read_binary_images("digits_data/train_images")
train_images = np.reshape(train_images, (-1, 28, 28))

test_labels = read_binary_labels("digits_data/test_labels")
test_images = read_binary_images("digits_data/test_images")
test_images = np.reshape(test_images, (-1, 28, 28))

train_labels_vector = np.reshape(train_labels, (-1, 1))
test_labels_vector = np.reshape(test_labels, (-1, 1))


def batch_generator(images, labels, batch_size, shuffle):
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)
    for num in range(0, len(images), batch_size):
        batch_indices = indices[num : num + batch_size]
        yield images[batch_indices], labels[batch_indices]


def forward_linear(x, shape):  # the shape can be (784, 128) 784 pixels and 128 rows
    bias = np.zeros(shape)
    weight = np.random.uniform(-0.1, 0.1, size=shape)
    return np.dot(prev, W) + b


def softmax(logits):
    exps = np.exp(logits)
    probs = exps / np.sum(exps)
    return probs


def cross_entropy_loss(predictions, targets):
    num_samples = len(predictions)
    correct_confidences = predictions[range(num_samples), targets]
    negative_log_likelihoods = -np.log(correct_confidences)
    loss = np.sum(negative_log_likelihoods) / num_samples
    return loss


def linear_backward(dZ2, A1, W2):
    m = A1.shape[0]
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2)
    dZ1 = np.dot(dZ2, W.T)
    return dZ1, dW2, db2
