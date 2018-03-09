# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 3c:
# Building the Recurrent Network Model
# ==============================================================================

import numpy as np
import tensorflow as tf
# `rnn` module temporarily in contrib. It's moving back to code in TF 1.1.
from tensorflow.contrib import layers, rnn

from util import ALPHABET_SIZE, read_data_files, rnn_minibatch_generator

# A. Load Training Data
# ------------------------------------------------------------------------------
training_data, validation_data, file_index = read_data_files(
    '../data/shakespeare/*',
    validation=True)

print(f'Num training characters: {len(training_data)}')
print(f'Num test/validation characters: {len(validation_data)}')
print(f'Num text files processed: {len(file_index)}')


# B. Build RNN Model
# ------------------------------------------------------------------------------

# Configurable hyperparameters.
BATCH_SIZE = 200
SEQUENCE_LENGTH = 30
NUM_HIDDEN_LAYERS = 3
GRU_INTERNAL_SIZE = 512

# *** Inputs (sequence of characters that are encoded as ints)

# [BATCH_SIZE, SEQUENCE_LENGTH]
X = tf.placeholder(tf.uint8, [None, None], name='X')
# [BATCH_SIZE, SEQUENCE_LENGTH, APLHABET_SIZE]
Xo = tf.one_hot(X, ALPHABET_SIZE, 1.0, 0.0)
# [BATCH_SIZE, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS]
H_in = tf.placeholder(
    tf.float32,
    [None, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS],
    name='Hin')

# Expected outputs (sequence of characters encoded as ints).
# Define expected RNN outputs. This is used for training.
# This is the same sequence as the input sequence, but shifted by 1 since we are
# trying to predict the next character.

# [BATCH_SIZE, SEQUENCE_LENGTH]
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y')
# [BATCH_SIZE, SEQUENCE_LENGTH, APLHABET_SIZE]
Yo_ = tf.one_hot(Y_, ALPHABET_SIZE, 1.0, 0.0)


# *** Internal/hidden RNN layers.

# The RNN is composed of a certain number of hidden layers, where each node is
# `GruCell` that uses `GRU_INTERNAL_SIZE` as the internal state size of a
# single cell. A higher `GRU_INTERNAL_SIZE` means more complex state can be
# stored in a single cell, at the cost of more computation.
cells = [rnn.GRUCell(GRU_INTERNAL_SIZE) for _ in range(NUM_HIDDEN_LAYERS)]
multicell = rnn.MultiRNNCell(cells, state_is_tuple=False)

# Using `dynamic_rnn` means Tensorflow "performs fully dynamic unolling" of the
# network. This is faster than compiling the full graph at initialisation time.
# TODO: verify
#
# Yr:    [ BATCH_SIZE, SEQUENCE_LENGTH, GRU_INTERNAL_SIZE ]
# H_out: [ BATCH_SIZE, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS ]
#
# # NOTE: H_out is the last state in the sequence.
Yr, H_out = tf.nn.dynamic_rnn(
    multicell,           # hidden cell state layers
    Xo,                  # input characters
    initial_state=H_in,  # input cell state (set to output state of last run)
    dtype=tf.float32)

# H_out is the output cell state after running a time step of the recurrent
# network. We wrap it in a tf object just to give it a name in the computation
# graph (we'll see why that's useful later).
H_out = tf.identity(H_out, name='H_out')


# *** Outputs
# Softmax layer implementation:
# Flatten the first two dimensions of the output before applying softmax.
# This performs the following transformation:
#
# [ BATCH_SIZE, SEQUENCE_LENGTH, ALPHABET_SIZE ]
#     => [ BATCH_SIZE x SEQUENCE_LENGTH, ALPHABET_SIZE ]
Yflat = tf.reshape(Yr, [-1, GRU_INTERNAL_SIZE])

# After this transformation, apply softmax readout layer. This way, the weights
# and biases are shared across unrolled time steps. From the readout point of
# view, a value coming from a cell or a minibatch is the same thing.
Ylogits = layers.linear(Yflat, ALPHABET_SIZE)  # [ BATCH_SIZE x SEQUENCE_LENGTH, ALPHABET_SIZE ]
Yo = tf.nn.softmax(Ylogits, name='Yo')         # [ BATCH_SIZE x SEQUENCE_LENGTH, ALPHABET_SIZE ]
Y = tf.argmax(Yo, 1)                           # [ BATCH_SIZE x SEQUENCE_LENGTH ]
Y = tf.reshape(Y, [BATCH_SIZE, -1], name='Y')  # [ BATCH_SIZE, SEQUENCE_LENGTH ]

# Final outputs are: Y and H_out


# C. Computing Loss and Optimising It
# ------------------------------------------------------------------------------

# *** Loss and Optimiser

# Need to also flatten the expected one-hot char outputs, in order to compute
# softmax loss.
Yflat_ = tf.reshape(Yo_, [-1, ALPHABET_SIZE])  # [ BATCH_SIZE x SEQUENCE_LENGTH, ALPHABET_SIZE ]

# TODO: some notes on this loss function
loss = tf.nn.softmax_cross_entropy_with_logits(  # [ BATCH_SIZE x SEQUENCE_LENGTH ]
    logits=Ylogits,
    labels=Yflat_)
loss = tf.reshape(loss, [BATCH_SIZE, -1])        # [ BATCH_SIZE, SEQUENCE_LENGTH ]

# Used to adjust the weights at each training step to minimize the loss
# function.
train_step = tf.train.AdamOptimizer().minimize(loss)

# Add extra graph outputs that contain various stats about the loss and accuracy
# of the model.
#
# These are used purely for our own visual evaluation of the quality of the
# model. It is not required to train the model (only the loss is required for
# that).
sequence_loss = tf.reduce_mean(loss, 1)
batch_loss = tf.reduce_mean(sequence_loss)
batch_accuracy = tf.reduce_mean(
    tf.cast(
        # Y_ = expected next char, Y = predicted next char
        tf.equal(Y_, tf.cast(Y, tf.uint8)),
        tf.float32))
