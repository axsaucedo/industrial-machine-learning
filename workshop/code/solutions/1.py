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
inputs = tf.placeholder(tf.float32, [2])
# Graph Node 2: an internal operation. Multiply both inputs by 3.
multiplied_inputs = tf.scalar_mul(3, inputs)
# Graph Node 3: final output. Sum the multiplied inputs.
output_sum = tf.reduce_sum(multiplied_inputs)

# Setup to visualise the generated graph using Tensorboard.
session = tf.Session()
summary_writer = tf.summary.FileWriter(
    os.path.join('logs', 'example'),
    graph=session.graph)

# Run the graph.
result = session.run(output_sum, feed_dict={inputs: [10, 15]})
print(result)
