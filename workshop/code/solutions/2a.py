# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 2a:
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
