# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 2b:
# Building a Basic Neural Network
# ==============================================================================

import csv
import os
from typing import Generator, Iterable, Tuple

import numpy as np
import tensorflow as tf


# A. Load Training Data
# ------------------------------------------------------------------------------
CLASS_MAPPING = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
}


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray]:
    filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'iris.csv')
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)

        input_feature_vectors = []
        expected_outputs = []
        for row in reader:
            # Extract four input features and convert to floats
            features = [float(row[i]) for i in range(4)]
            input_feature_vectors.append(features)
            # Convert string output classes to integers (required to use as
            # output in neural networks).
            output_class = CLASS_MAPPING[row[4]]
            expected_outputs.append([output_class])

    return np.array(input_feature_vectors), np.array(expected_outputs)


# Split dataset into inputs and expected outputs
inputs, expected_outputs = load_iris_dataset()
print(f'Training data points: {len(inputs)}')


# B. Building Neural Network Tensorflow Model
# ------------------------------------------------------------------------------

# Model hyper parameters
NUM_INPUTS = 4
NUM_CLASSES = len(CLASS_MAPPING)
NUM_HIDDEN_LAYER_NODES = 6
LEARNING_RATE = 0.001

# Input features
x = tf.placeholder("float", [None, NUM_INPUTS])
# Expected output
y = tf.placeholder("int32", [None, 1])
yo = tf.one_hot(y, NUM_CLASSES, 1.0, 0.0)

# Build NN model computation graph
weights = {
    'hidden': tf.Variable(
        tf.random_normal([NUM_INPUTS, NUM_HIDDEN_LAYER_NODES])),
    'output': tf.Variable(
        tf.random_normal([NUM_HIDDEN_LAYER_NODES, NUM_CLASSES]))
}
biases = {
    'hidden': tf.Variable(
        tf.random_normal([NUM_HIDDEN_LAYER_NODES])),
    'output': tf.Variable(
        tf.random_normal([NUM_CLASSES]))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])


logits_output_layer = tf.add(
    tf.matmul(hidden_layer, weights['output']), biases['output'])

softmax_output_layer = tf.nn.softmax(logits_output_layer, 1)
prediction = tf.argmax(softmax_output_layer, 1)

# Define loss function (to measure predictive power of network) and create a
# node that optimises the weights to minimise the loss function.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_output_layer,
    labels=yo))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_operation = optimizer.minimize(loss)
