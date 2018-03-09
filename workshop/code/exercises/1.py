# ==============================================================================
# Deep Learning with Recurrent Neural Networks Workshop
# By Donald Whyte and Alejandro Saucedo
#
# Step 1:
# Tensorflow Hello World
# ==============================================================================

import os

import tensorflow as tf

# Build computation graph.

# Graph Node 1: two input floats.
# TODO: pass in data type and vector size
inputs = tf.placeholder(TODO)
# Graph Node 2: an internal operation. Multiply both inputs by 3.
# TODO: pass in coefficent to multiply by, vector to multiply)
multiplied_inputs = tf.scalar_mul(TODO)
# Graph Node 3: final output. Sum the multiplied inputs.
# TODO: pass in vector whose elements we want to sum
output_sum = tf.reduce_sum(TODO)

# Setup to visualise the generated graph using Tensorboard.
session = tf.Session()
summary_writer = tf.summary.FileWriter(
    os.path.join('log', 'example'),
    graph=session.graph)

# TODO: run the following command to open Tensorboard and view the graph you've created:
#     tensorboard --logdir log

# Run the graph.
result = session.run(
    # pass in node that you want to compute (the output) here:
    TODO,
    feed_dict={
        inputs: [
            # TODO: your two input numbers here
        ]
    }
)

print(result)
