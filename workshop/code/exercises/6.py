# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 6:
# Using Saved Models to Generate Text
# ==============================================================================

import numpy as np
import tensorflow as tf

from util import (convert_from_alphabet,
                  convert_to_alphabet_int,
                  convert_to_alphabet_str,
                  sample_from_probabilities)


# Saved model to load.
META_GRAPH_FILE = 'checkpoints/rnn_train_1508428333-0.meta'
WEIGHTS_FILE = 'checkpoints/rnn_train_1508428333-0'
# These must match what was saved in the model's checkpoint!
NUM_HIDDEN_LAYERS = 3
GRU_INTERNAL_SIZE = 512


class CharPrinter:

    def __init__(self, max_line_length: int=100):
        self._max_line_length = max_line_length
        self._num_chars_on_current_line = 0

    def print_char(self, ch: str):
        print(ch, end='')
        if ch == '\n':
            self._num_chars_on_current_line = 0
        else:
            self._num_chars_on_current_line += 1
            if self._num_chars_on_current_line == self._max_line_length:
                print('')
                self._num_chars_on_current_line = 0


initial_char = convert_from_alphabet(ord('L'))
printer = CharPrinter()
with tf.Session() as session:
    # Load saved model
    new_saver = tf.train.import_meta_graph(META_GRAPH_FILE)
    new_saver.restore(session, WEIGHTS_FILE)

    # Set initial inputs.
    # [BATCH_SIZE, SEQUENCE_LENGTH] with BATCH_SIZE=1 and SEQUENCE_LENGTH=1
    x = np.array([[initial_char]])

    # Set initial hidden state to zeros.
    # [ BATCHSIZE, INTERNALSIZE * NUM_HIDDEN_LAYERS]
    h = np.zeros(
        [1, GRU_INTERNAL_SIZE * NUM_HIDDEN_LAYERS],
        dtype=np.float32)

    for i in range(1000000000):
        yo, h = session.run(
            ['Yo:0', 'H_out:0'],
            feed_dict={
                'X:0': x,
                'Hin:0': h,
                'batch_size:0': 1
            })

        # If sampling is be done from the topn most likely characters, the generated
        # text is more credible and more "english". If topn is not set, it defaults
        # to the full distribution (ALPHABET_SIZE).

        # Recommended:
        #     topn = 10 for intermediate checkpoints,
        #     topn = 2 or 3 for fully trained checkpoints
        ch = sample_from_probabilities(yo, topn=2)

        printer.print_char(convert_to_alphabet_str(ch))

        # Set next input char to the char that was just predicted
        x = np.array([[convert_to_alphabet_int(ch)]])
