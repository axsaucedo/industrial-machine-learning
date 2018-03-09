# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 3a:
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
