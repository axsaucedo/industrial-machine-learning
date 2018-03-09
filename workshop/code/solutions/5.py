# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 5:
# Saving Models
# ==============================================================================

import math
import os
import time

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
SEQUENCE_LENGTH = 30
NUM_HIDDEN_LAYERS = 3
GRU_INTERNAL_SIZE = 512

# *** Inputs (sequence of characters that are encoded as ints)

# Make the batch_size a configurable hyper-parameter. This allows us to run
# the training and validation steps with different batch sizes.
batch_size = tf.placeholder(tf.int32, name='batch_size')

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
Y = tf.reshape(Y, [batch_size, -1], name='Y')  # [ BATCH_SIZE, SEQUENCE_LENGTH ]

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
loss = tf.reshape(loss, [batch_size, -1])        # [ BATCH_SIZE, SEQUENCE_LENGTH ]

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

loss_summary = tf.summary.scalar('batch_loss', batch_loss)
acc_summary = tf.summary.scalar('batch_accuracy', batch_accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])



# D. Training RNN Model
# ------------------------------------------------------------------------------

# Configurable hyperparameters.
NUM_EPOCHS = 3
TRAINING_BATCH_SIZE = 200
STEP_SIZE = TRAINING_BATCH_SIZE * SEQUENCE_LENGTH
DISPLAY_ACC_EVERY_N_BATCHES = 5
VALIDATE_EVERY_N_BATCHES = 20
DISPLAY_ACC_EVERY_N_STEPS = DISPLAY_ACC_EVERY_N_BATCHES * STEP_SIZE
VALIDATE_EVERY_N_STEPS = VALIDATE_EVERY_N_BATCHES * STEP_SIZE
# --NEW--
SAVE_WEIGHTS_EVERY_N_BATCHES = 20
SAVE_WEIGHTS_EVERY_N_STEPS = SAVE_WEIGHTS_EVERY_N_BATCHES * STEP_SIZE
# --NEW--

# Initialise input hidden cell state to all zeros.
input_state = np.zeros([TRAINING_BATCH_SIZE, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS])

# Split training data into mini-batches and iterate over all batches
# `NUM_EPOCHS` times.
batch_sequencer = rnn_minibatch_generator(
    training_data,
    TRAINING_BATCH_SIZE,
    SEQUENCE_LENGTH,
    NUM_EPOCHS)

# Construct session and initialise all Tensorflow graph variables.
# We never defined any `tf.Variable`s explictly in the code, but the utility
# functions we're using (e.g. `rnn.GRUCell`) do.
session = tf.Session()
var_initializer = tf.global_variables_initializer()
session.run(var_initializer)

# Init Tensorboard stuff. This will save Tensorboard information into a
# different folder at each run named 'log/<timestamp>/'. Two sets of data are
# saved so that you can compare training and validation curves visually in
# Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter(f'log/{timestamp}-training')
summary_writer.add_graph(session.graph)
validation_writer = tf.summary.FileWriter(f'log/{timestamp}-validation')

# --NEW--
# Init for saving models. They will be saved into a directory named
# 'checkpoints'. Only the last checkpoint is kept.
if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')
saver = tf.train.Saver(max_to_keep=1000)  # TODO: what's this arg?
# --NEW--

step = 0
for batch_x, batch_y, current_epoch in batch_sequencer:
    # Train on one minibatch.
    step_inputs = {
        X: batch_x,
        Y_: batch_y,
        H_in: input_state,
        batch_size: TRAINING_BATCH_SIZE
    }

    # This is the call that actually trains on the batch. We pass in
    # the Tensorflow graph nodes we wish to compute the output of.
    #
    # Here, we want to run the training step (but don't care about its output),
    # and get modified internal cell state after running the training step.
    _, output_state = session.run([train_step, H_out], feed_dict=step_inputs)

    # Every so often, calculate batch/accuracy loss and display it to the user.
    step_should_display_accuracy = (
        step % DISPLAY_ACC_EVERY_N_STEPS == 0)
    if step_should_display_accuracy:
        step_loss, step_acc, step_summaries = session.run(
            [batch_loss, batch_accuracy, summaries],
            feed_dict=step_inputs)
        print(
            f'Epoch {current_epoch}, '
            f'Step {step}, '
            f'Minibatch Loss={step_loss:.4f}, '
            f'Minibatch Accuracy={step_acc * 100:.2f}, ')
        summary_writer.add_summary(step_summaries, step)

    # Every so often, calculate the loss/accuracy when using the model to
    # predict the validation sequence.
    should_validate = step % VALIDATE_EVERY_N_STEPS == 0
    if should_validate:
        print('Running current model on validation data')

        # Sequence length for validation. State will be wrong at the start of
        # each sequence.
        VALIDATION_SEQUENCE_LEN = 1 * 1024
        vali_batch_size = len(validation_data) // VALIDATION_SEQUENCE_LEN
        #txt.print_validation_header(len(codetext), bookranges)
        vali_x, vali_y, _ = next(rnn_minibatch_generator(  # all data in 1 batch
            validation_data,
            vali_batch_size,
            VALIDATION_SEQUENCE_LEN,
            num_epochs=1))

        vali_nullstate = np.zeros(
            [vali_batch_size, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS])
        vali_inputs = {
            X: vali_x,
            Y_: vali_y,
            H_in: vali_nullstate,
            batch_size: vali_batch_size
        }
        vali_loss, vali_acc, vali_summaries = session.run(
            [batch_loss, batch_accuracy, summaries],
            feed_dict=vali_inputs)
        print(
            f'Epoch {current_epoch}, '
            f'Step {step}, '
            f'Validation Loss={vali_loss:.4f}, '
            f'Validation Accuracy={vali_acc * 100:.2f}, ')
        validation_writer.add_summary(vali_summaries, step)

    # --NEW--
    # Every so often, save the current model weights to a file.
    should_save_checkpoint = step % SAVE_WEIGHTS_EVERY_N_STEPS == 0
    if should_save_checkpoint:
        print('Saved model weights')
        saved_file = saver.save(
            session,
            f'checkpoints/rnn_train_{timestamp}',
            global_step=step)
        print(f'Saved file: {saved_file}')
    # --NEW--

    # IMPORTANT: Use the output state as the input state in the next training
    # step.
    input_state = output_state
    step += STEP_SIZE
