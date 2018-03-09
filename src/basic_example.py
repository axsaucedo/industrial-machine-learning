import os

import tensorflow as tf

# Graph Node 1: inputs
inputs = tf.placeholder(tf.float32, [2])
# Graph Node 2: an internal operation
multiplied_inputs = tf.scalar_mul(3, inputs)
# Graph Node 3: final output
output_sum = tf.reduce_sum(multiplied_inputs)

# Setup to visualise the graph.
session = tf.Session()
summary_writer = tf.summary.FileWriter(
    os.path.join('logs', 'example'),
    graph=session.graph)

# Run the graph.
result = session.run(output_sum, feed_dict={inputs: [10, 15]})
print(result)
