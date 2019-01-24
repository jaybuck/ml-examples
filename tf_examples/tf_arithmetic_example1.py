#!/usr/bin/env python
"""
Very simple example of how to use TensorFlow.

It builds a computation graph for a very simple example:
res = (a * b) / (a + b)

where:
a = 15
b = 5
"""

import tensorflow as tf
import numpy as np

a = tf.constant(15, name="a")
b = tf.constant(5, name="b")

prod = tf.multiply(a, b, name="mult")
sum = tf.add(a, b, name="add")

res = tf.divide(prod, sum, name="divide")

# Initialize all variables. Do this first in upcoming session.
init = tf.initializers.global_variables()

# Launch the TensorFlow session used to run the graph:
sess = tf.Session()
sess.run(init)

# Use the run function to run the graph and get values from the run.
outval, a_out, b_out = sess.run([res, a, b])
print('a: {}\tb: {}\tresult: {}'.format(a_out, b_out, outval))

