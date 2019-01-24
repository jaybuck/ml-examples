#!/usr/bin/env python
"""
Very simple example of how to use TensorFlow.

It builds a computation graph for a very simple example:
res = (a * b) / (a + b)

where:
a = 15
b = 5

In this version we add commands to do logging for TensorBoard.
In order to make the TensorBoard output more interesting, we also:
* Use tf.placeholder() for the inputs so they show up on TensorBoard.
* Add an op that compares the result of the above computation to a target in order to compute an error.
* Do sess.run() in a loop, incrementing the inputs, so that TensorBoard can plot some graphs of values as a function of epoch.
"""

import os
import argparse

import tensorflow as tf
import numpy as np

homedir = os.path.expanduser('~')

parser = argparse.ArgumentParser(description='Example of simple scalar arithmetic in TensorFlow')

# Optional command-line args
parser.add_argument('-a', '--a_value', type=float, default=15.0,
                    help='First input value')
parser.add_argument('-b', '--b_value', type=float, default=5.0,
                    help='Second input value')
parser.add_argument('-y', '--target', type=float, default=0.0,
                    help='Target result')
parser.add_argument('-s', '--summaries_dir', default='arithmetic_example_logs')

args = parser.parse_args()
a_val = args.a_value
b_val = args.b_value
target_result = args.target
summaries_dir = os.path.join(homedir, args.summaries_dir)

# Clean out summaries dir
if tf.gfile.Exists(summaries_dir):
    print("Deleting {0}".format(summaries_dir))
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir)

with tf.name_scope('input'):
    a = tf.placeholder(tf.float32, shape=(), name='a')
    b = tf.placeholder(tf.float32, shape=(), name='b')
    y = tf.placeholder(tf.float32, shape=(), name='y')

with tf.name_scope("mulitply"):
    prod = tf.multiply(a, b, name="mult")
    tf.summary.scalar('prod', prod)

with tf.name_scope('addition'):
    sum1 = tf.add(a, b, name="add")
    tf.summary.scalar('sum1', sum1)

with tf.name_scope('result'):
    res = tf.divide(prod, sum1, name="divide")
    tf.summary.scalar('res', res)

# Evaluate "accuracy"
with tf.name_scope('error'):
    error = tf.squared_difference(y, res, name='error')
    tf.summary.scalar('error', error)

# Initialize all variables. Do this first in upcoming session.
init = tf.initializers.global_variables()

# Launch the TensorFlow session used to run the graph:
sess = tf.Session()

# Merge all summaries and write them out to summaries_dir
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

sess.run(init)

# Use the run function to run the graph and get values from the run.

for epoch in range(8):
    summary, err, result_val, a_out, b_out = sess.run([merged, error, res, a, b],
                                                      feed_dict={a: a_val + epoch, b: b_val + epoch, y: target_result})
    train_writer.add_summary(summary, epoch)
    print('a: {}\tb: {}\ttarget: {}\tresult: {}\terror {}'.format(a_out, b_out, target_result, result_val, err))
