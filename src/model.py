import datetime
import logging
import math
import numpy as np
import os
import time
from typing import Callable, List, Optional

import tensorflow as tf
from tensorflow.contrib import layers
# `rnn` module temporarily in contrib. It's moving back to code in TF 1.1.
from tensorflow.contrib import rnn

from text import (ALPHABET_SIZE,
                  print_learning_learned_comparison,
                  rnn_minibatch_generator,
                  SequenceData)

_logger = logging.getLogger(__name__)


class RNNTextModel:

    def __init__(self,
                 sequence_length: int,
                 batch_size: int,
                 gru_internal_size: int,
                 num_hidden_layers: int,
                 stats_log_dir: str):
        """Construct `RNNTextModel` with specified hyperparameters.

        This model is a deep recurrent neural network that uses GRU cells for
        long-term memory.

        `sequence_length` is the length of each input sequence. Longer
        sequences mean the model has longer-term memory, but networks that use
        longer sequences (and thus, have more time steps) are harder/take
        longer to learn.

        `batch_size` is the number of sequences to put in each mini-batch. The
        network's weights are onlt adjusted after each mini-batch.

        `gru_internal_size` specifies the number of nodes inside each hidden
        GRU cell layer.

        `num_hidden_layers` specifies the number of hidden layers (number of
        GRU cell lateys) to use in the deep RNN.

        The model's loss and graph will be periodically logged to the
        `stats_log_dir` directory.
        """
        # Mark time this text model was initially created
        self._timestamp = str(math.trunc(time.time()))

        # Store hyperparameters
        self._sequence_length = sequence_length
        self._batch_size = batch_size
        self._gru_internal_size = gru_internal_size
        self._num_hidden_layers = num_hidden_layers

        # ---------------------------------------------------------------------
        # Graph Inputs
        # ---------------------------------------------------------------------
        X = tf.placeholder(tf.uint8, [None, None], name='X')
        self._inputs = {
            # Hyperparameters.
            'learning_rate': tf.placeholder(tf.float32, name='learning_rate'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),

            # Dimensions: [ batch_size, sequence_length ]
            'X': X,
            # Dimensions: [ batch_size, sequence_length, ALPHABET_SIZE ]
            'Xo': tf.one_hot(X, ALPHABET_SIZE, 1.0, 0.0),

            # Input cell state.
            # Dimensions: [batch_size, gru_internal_size * num_hidden_layers]
            'H_in': tf.placeholder(
                tf.float32,
                [None, self._gru_internal_size * self._num_hidden_layers],
                name='Hin')
        }

        # Define expected RNN outputs. This is used for training.
        # This is the same sequence as the input sequence, but shifted by 1
        # since we are trying to predict the next character.
        Y_exp =  tf.placeholder(tf.uint8, [None, None], name='Y_exp')
        self._expected_outputs = {
            # Dimensions: [ batch_size, sequence_length ]
            'Y': Y_exp,
            # Dimensions: [ batch_size, sequence_length, ALPHABET_SIZE ]
            'Yo': tf.one_hot(Y_exp, ALPHABET_SIZE, 1.0, 0.0)
        }

        # ---------------------------------------------------------------------
        # Hidden Layers
        # ---------------------------------------------------------------------

        # Define internal/hidden RNN layers. The RNN is composed of a certain
        # number of hidden layers, where each node is `GruCell` that uses
        # `gru_internal_size` as the internal state size of a single cell. A
        # higher `gru_internal_size` means more complex state can be stored in
        # a single cell.
        self._cells = [
            rnn.GRUCell(self._gru_internal_size)
            for _ in range(self._num_hidden_layers)
        ]
        self._multicell = rnn.MultiRNNCell(self._cells, state_is_tuple=False)

        # Using `dynamic_rnn` means Tensorflow "performs fully dynamic
        # unrolling" of the network. This is faster than compiling the full
        # graph at initialisation time.
        #
        # Note that compiling the full grapgh at train time isn’t that big of
        # an issue for training, because we only need to build the graph once.
        # It could be a big issue, however, if we need to build the graph
        # multiple times at test time. And remember, this training loop does
        # occassionally process inputs via test time, through the occassional
        # reports it outputs.
        #
        # Yr: [ batch_size, sequence_length, gru_internal_size ]
        # H:  [ batch_size, gru_internal_size * num_hidden_layers ]
        # H_out is the last state in the sequence.
        Yr, H_out = tf.nn.dynamic_rnn(
            self._multicell,
            self._inputs['Xo'],
            dtype=tf.float32,
            initial_state=self._inputs['H_in'])

        # ---------------------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------------------

        # Softmax layer implementation:
        # Flatten the first two dimensions of the output. This performs the
        # following transformation:
        #
        # [ batch_size, sequence_length, ALPHABET_SIZE ]
        #     => [ batch_size x sequence_length, ALPHABET_SIZE ]
        Yflat = tf.reshape(Yr, [-1, self._gru_internal_size])

        # After this transformation, apply softmax readout layer. This way, the
        # weights and biases are shared across unrolled time steps. From the
        # readout point of view, a value coming from a cell or a minibatch is
        # the same thing.
        Ylogits = layers.linear(Yflat, ALPHABET_SIZE)                  # [ batch_size x sequence_length, ALPHABET_SIZE ]
        Yflat_ = tf.reshape(                                           # [ batch_size x sequence_length, ALPHABET_SIZE ]
            self._expected_outputs['Yo'], [-1, ALPHABET_SIZE])
        Yo = tf.nn.softmax(Ylogits, name='Yo')                         # [ batch_size x sequence_length, ALPHABET_SIZE ]
        Y = tf.argmax(Yo, 1)                                           # [ batch_size x sequence_length ]
        Y = tf.reshape(Y, [self._inputs['batch_size'], -1], name='Y')  # [ batch_size, sequence_length ]

        # Store the output nodes in a dictionary for easy access later.
        self._outputs = {
            'Y': Y,
            # Output cell state after running a time step of the recurrent
            # network. We specify this just to give H_out a identifiable name.
            'H_out': tf.identity(H_out, name='H_out')
        }

        # Commpute the loss (error) of the network.
        self._loss = tf.nn.softmax_cross_entropy_with_logits(          # [ batch_size x sequence_length ]
            logits=Ylogits, labels=Yflat_)
        self._loss = tf.reshape(                                       # [ batch_size, sequence_length ]
            self._loss, [self._inputs['batch_size'], -1])

        # Used to adjust the weights at each training step, sich that the
        # loss function is minimised.
        self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

        # Stats not used to directly train the network, but are logged so they
        # can be viewed by the human user.
        self._sequence_loss = tf.reduce_mean(self._loss, 1)
        self._batch_loss = tf.reduce_mean(self._sequence_loss)
        self._batch_accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    self._expected_outputs['Y'],
                    tf.cast(self._outputs['Y'], tf.uint8)),
                tf.float32))

        self._initialise_tf_session()
        self._build_statistics(stats_log_dir)

    def _build_statistics(self, stats_log_dir: str):
        loss_summary = tf.summary.scalar('batch_loss', self._batch_loss)
        acc_summary = tf.summary.scalar('batch_accuracy', self._batch_accuracy)
        self._summaries = tf.summary.merge([loss_summary, acc_summary])

        self._stats_summary_writer = tf.summary.FileWriter(
            os.path.join(stats_log_dir, f'{self._timestamp}-training'),
            graph=self._session.graph)
        self._validation_stats_writer = tf.summary.FileWriter(
            os.path.join(stats_log_dir, f'{self._timestamp}-validation'))

    def _initialise_tf_session(self):
        initalised_vars = tf.global_variables_initializer()
        self._session = tf.Session()
        self._session.run(initalised_vars)
        self._initialised = True

    def run_training_loop(self,
                          data: SequenceData,
                          num_epochs: int,
                          learning_rate: float,
                          display_every_n_batches: int,
                          checkpoint_dir: str,
                          on_step_complete: Callable[[int], None],
                          should_stop: Callable[[int], bool]):
        """Run training loop on the model object.

        `data` contains the training and test text sequences (encoded as
        integers for dircect input in the neural network).

        `num_epochs` and `learning_rate` specify the number of training epochs
        (iterations) and speed of learning respectively. If `learning_rate` is
        too low, then network will take a very long time to train. If it is too
        high, then the network will not converge on a good set of weights.

        `checkpoint_dir` specifies the directory the trained model will be
        periodically saved to.

        `display_every_n_batches` specifies the number of batches that should
        run before displaying the network's current loss and example output.

        `on_step_complete` is called when a training step is complete. Users
        can implement their own logic for when a step is completed, or pass in
        an empty function to do nothing after each step.

        `should_stop` is called after every step. It should return `True` if
        the training loop should continue running and `False` if it should not.
        This is used to implement custom stopping criteria.
        """
        training_text, test_text, file_index = data

        # We display information on the currently trained model's accuracy
        # after training every N batches. Each batch corresponds to a single
        # step of the training loop. Each batch processes a number of sequences
        # which are X characters in length. Therefore, a step's size is defined
        # as the number of input sequences processed in a batch multiplied by
        # _length_ of each input sequence.
        step_size = self._batch_size * self._sequence_length
        display_every_n_batches = display_every_n_batches * step_size

        # Create directory that stores checkpoints of the trained model.
        # Training models can take a very long time (several hours or days), so
        # storing intermediate results on disk prevents us losing all progress
        # if the process crashes.
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        saver = tf.train.Saver(max_to_keep=1)

        # Initialise input cell state to all zeros.
        input_state = np.zeros([
            self._batch_size,
            self._gru_internal_size * self._num_hidden_layers,
        ])

        # Split training data into mini-batches and iterate over all batches
        # `num_epochs` times.
        batch_sequencer = rnn_minibatch_generator(
            training_text,
            self._batch_size,
            self._sequence_length,
            num_epochs)

        step = 0
        for x, y_, epoch in batch_sequencer:
            # Train on one minibatch.
            feed_dict = {
                self._inputs['learning_rate']: learning_rate,
                self._inputs['batch_size']: self._batch_size,
                self._inputs['X']: x,
                self._inputs['H_in']: input_state,
                self._expected_outputs['Y']: y_
            }

            # This is the call that actually trains on the batch. We pass in
            # the Tensorflow graph nodes we wish to compute the output of.
            _, y, output_state, summary = self._session.run(
                [
                    self._train_step,
                    self._outputs['Y'],
                    self._outputs['H_out'],
                    self._summaries
                ],
                feed_dict=feed_dict)

            # Save training data for Tensorboard
            self._stats_summary_writer.add_summary(summary, step)

            # Run a validation step every 50 batches.
            # The validation text should be a single sequence but that's too
            # slow (1s per 1024 chars!), so we cut it up and batch the pieces
            # (slightly inaccurate).
            #
            # tested: validating with 5K sequences instead of 1K is only
            # slightly more accurate, but a lot slower.
            if step % display_every_n_batches == 0:
                # We don't use dropout for running the validation!
                feed_dict = {
                    self._inputs['X']: x,
                    self._expected_outputs['Y']: y_,
                    self._inputs['batch_size']: self._batch_size,
                    self._inputs['H_in']: input_state
                }
                y, losses, batch_loss, batch_accuracy = self._session.run(
                    [
                        self._outputs['Y'],
                        self._sequence_loss,
                        self._batch_loss,
                        self._batch_accuracy
                    ],
                    feed_dict=feed_dict)
                epoch_size = len(training_text) // (
                    self._batch_size * self._sequence_length)
                print_learning_learned_comparison(
                    x[:5],
                    y,
                    losses,
                    file_index,
                    batch_loss,
                    batch_accuracy,
                    epoch_size,
                    step,
                    epoch)
                # TODO: do proper validation!

            # Save a checkpoint (every 500 batches)
            if step // 10 % display_every_n_batches == 0:
                saver.save(
                    self._session,
                    f'checkpoints/rnn_train_{self._timestamp}',
                    global_step=step)

            on_step_complete(step)
            if should_stop(step):
                break

            # Loop state around
            input_state = output_state
            step += step_size
