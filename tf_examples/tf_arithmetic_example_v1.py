#!/usr/bin/env python
"""
Very simple example of how to use TensorFlow.

It shows how one can do the matrix multiply of an input vector and a weight vector.

It:
- Initializes two vectors
  - One via tf.constant()
  - One via tf.get_variable()
- Sum them
- Multiplies them together element-wise.
- Does a dot product via matmul()
"""

import tensorflow as tf
import numpy as np

# Input variable. Constant for this simple example
x = tf.constant([[1.1, 2.2, 3.3, 4.4]], name='x', dtype=tf.float32)

# Weight vector. A variable
w = tf.get_variable('w', shape=(4, 1), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

# Compute the sum of x and w:
sum_x_w = tf.add(x, tf.transpose(w), name='sum_x_w')

# Compute the element-wise multiply of x and w:

mult_x_w = tf.multiply(x, tf.transpose(w), name='mult_x_w')

# Compute the dot product of x and w, then add bias b:
wsum = tf.matmul(x, w, name='wsum')

# Initialize all variables. Do this first in upcoming session.
init = tf.initializers.global_variables()

# Launch the TensorFlow session used to run the graph:
sess = tf.Session()
sess.run(init)

# Use the run function to run the graph and get values from the run.
wsum_out, mult_out, sum_out, x_out, w_out = sess.run([wsum, mult_x_w, sum_x_w, x, w])
print('x: {}\tshape: {}\n'.format(x_out, x_out.shape))
print('w: {}\tshape: {}\n'.format(w_out, w_out.shape))
print('sum_x_w: {}\tshape: {}\n'.format(sum_out, sum_out.shape))
print('mult_x_w: {}\tshape: {}\n'.format(mult_out, mult_out.shape))
print('weighted sum: {}\tshape: {}\n'.format(wsum_out, wsum_out.shape))
