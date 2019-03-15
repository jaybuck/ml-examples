#!/usr/bin/env python
# Simple TensorFlow classifier for the MNIST data.
# One layer of weights, then SoftMax

import os
import time
import argparse

import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='Logistic regression model for MNIST data using TensorFlow')

# Optional command-line args
parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Max number of epochs')
parser.add_argument('-b', '--batchsize', type=int, default=64,
                    help='Mini-batch size')

parser.add_argument('-l', '--learnrate', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('-d', '--datadir', default='data',
                    help='Directory holding MNIST data wrt your homedir')
parser.add_argument('-s', '--summariesdir', default='mnist_linregression_logs',
                    help='Summaries directory')

args = parser.parse_args()
nepochs = args.epochs
batch_size = args.batchsize
learnrate = args.learnrate

homedir = os.path.expanduser('~')
data_dir = os.path.join(homedir, args.datadir)
summaries_dir = os.path.join(homedir, args.summariesdir)

# How often to print training stats.
training_stats_interval = 100
if nepochs < 100:
    training_stats_interval = 10
elif nepochs > 80000:
    training_stats_interval = 1000

if nepochs < 20:
    training_stats_interval = 1

# Get MNIST data (if need be)
MNIST_DIR = data_dir
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

# Setup placeholder for input and correct output.
x = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

# Setup variables for weights as biases.
W = tf.Variable(tf.zeros([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# Implement the model.
logits = tf.matmul(x, W) + b
y = tf.nn.softmax(logits)

# Compute cross-entropy err:
# NB: This is a very naive approach to computing cross-entropy just to show the formula.
# It is not numerically stable so it is just to show what's going on.
cost_op = -tf.reduce_sum(y_ * tf.log(y))

# Graph node for training:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnrate)
train_step = optimizer.minimize(cost_op)

# Evaluate accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all variables.  Do this first in the upcoming session.
init = tf.initializers.global_variables()

# Launch the TensorFlow graph session:
sess = tf.Session()
sess.run(init)

# Do the training:

# Variables for timing stats:
n_batches = 0
batch_time_sum = 0.0

for epoch in range(nepochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    t0 = time.time()
    cost, train_accuracy, _ = sess.run([cost_op, accuracy_op, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    time_delta = time.time() - t0
    n_batches += 1
    batch_time_sum += time_delta

    if epoch % training_stats_interval == 0:
        test_accuracy = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("epoch {}\tcost {:.4f}\tTrainAccuracy {:.4f}\tTestAccuracy {:.4f}".format(epoch, cost, train_accuracy, test_accuracy))

# Timing stats
avg_train_time = batch_time_sum / (n_batches * batch_size)
print('Average train time per image: {}'.format(avg_train_time))


# Accuracy:
test_accuracy = sess.run(accuracy_op, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("\nepoch {}\tFinalTestAccuracy {:.4f}".format(epoch, test_accuracy))
