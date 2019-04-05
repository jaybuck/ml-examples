#!/usr/bin/env python
# Simple TensorFlow classifier for the MNIST data.
# Two convolutional layers.
# Two fully connected layers.
# SoftMax operation done on logit of final fully connected layer.
# The answer is the index of the element with the largest value in the SoftMax output.

import os
import os.path
import time
import argparse

import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")


def getActivations(tfsess, layer, stimuli):
    units = tfsess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})
    plotNNFilter(units)


if __name__ == '__main__':

    ################################################################################
    # Arg parsing
    #
    parser = argparse.ArgumentParser(description='CNN model for MNIST data using TensorFlow')

    # Optional command-line args
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Max number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('-r', '--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout')
    parser.add_argument('-l', '--learnrate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    parser.add_argument('-d', '--datadir', default='data',
                        help='Directory holding MNIST data wrt your homedir')
    parser.add_argument('-s', '--summariesdir', default='mnist_cnn_run_logs',
                        help='TensorBoard Summaries directory')
    parser.add_argument('-w', '--modeldir', default='models',
                        help='TensorBoard Summaries directory')


    args = parser.parse_args()
    max_train_steps = args.epochs
    batch_size = args.batchsize
    learnrate = args.learnrate
    dropoutrate = args.dropout

    homedir = os.path.expanduser('~')
    data_dir = os.path.join(homedir, args.datadir)
    summaries_dir = os.path.join(homedir, args.summariesdir)
    model_path = os.path.join(args.modeldir, 'mnistcnn.ckpt')

    # How often to print training stats.
    training_stats_interval = 100
    if max_train_steps < 100:
        training_stats_interval = 10
    elif max_train_steps > 80000:
        training_stats_interval = 1000

    if max_train_steps < 20:
        training_stats_interval = 1

    # Clean out summaries dir
    if tf.gfile.Exists(summaries_dir):
        print("Deleting {0}".format(summaries_dir))
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)

    # Get MNIST data (if need be)
    MNIST_DIR = os.path.join(data_dir, 'mnist')
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(MNIST_DIR, one_hot=True)

    # Setup placeholder for input and correct output.
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # First convolution layer:  32 units:
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        # Reshape x to make use of 2-D structure:
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # Do the convolution and use ReLU nonlinearity.
        # Then reduce size of output with a max pool layer down to 14 x 14
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolution layer: 64 units:
    # Size of max pool output is now down to 7 x 7
    with tf.name_scope('conv2'):
        # W_conv2 = weight_variable([5, 5, 32, 64])
        W_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer with 1024 units connected to conv2 layer:
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Use dropout to reduce overfitting:
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout layer: the answer.
    # Fully connected layer 2:  1024 input, 10 outputs: one for each digit.
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_hat = tf.nn.softmax(logits)

    # Use cross-entropy for loss measure:
    # Naive implementation: cost_op = -tf.reduce_sum(y_ * tf.log(y_conv))
    with tf.name_scope('cross_entropy'):
        cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
        tf.summary.scalar('cross_entropy', cost_op)

    # Use ADAM or GradientDescentOptimizer for gradient descent:
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learnrate)
        # optimizer = tf.train.GradientDescentOptimizer(learnrate)
        train_op = optimizer.minimize(cost_op)

    # Evaluate accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction_op = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction_op, tf.float32))
        tf.summary.scalar('accuracy', accuracy_op)

    # Initialize all variables.  Do this first in the upcoming session.
    init_op = tf.initializers.global_variables()

    # Add op to save and restore all of the variables.
    saver = tf.train.Saver()

    # Launch the TensorFlow graph session:
    sess = tf.Session()

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')

    sess.run(init_op)

    # Restore saved model.
    saver.restore(sess, model_path)
    print('Model restored from: {}'.format(model_path))
    # print('Computing test accuracy...')
    # print("\nfinal testaccuracy {:0.4f}".format(sess.run(accuracy_op,
    #                                                      feed_dict={x: mnist.test.images, y_: mnist.test.labels,
    #                                                                 keep_prob: 1.0})))

    # Test on one image
    indx = 3
    print("Hidden layer activations for test image {}".format(indx))
    testimg = mnist.test.images[indx]
    plt.imshow(np.reshape(testimg, [28, 28]), interpolation="nearest", cmap="gray")
    plt.show()

    getActivations(sess, h_conv2, testimg)

    plt.show()

