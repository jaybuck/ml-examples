#!/usr/bin/env python
# Simple TensorFlow classifier for the linear regression
# on 2D point data.
# One layer of weights, then logistic regression activation.

import sys
import time
import datetime
import argparse
import json

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def read_points_file(fname, delim=','):
    """
    Read x y points from a text file, returning numpy arrays containing the x,y coordinates and the target label.

    The file contains one row for each training example (m examples).
    The first field holds the target label for the example.
    The remaining fields hold the coordinates of the input vector. For this example,
    these are 2D points on the x0, x1 plane.
    We return two arrays: one for the x coordinate vectors and one for the label y values.
    The x ndarray has one column for each training example and 2 rows.
    The y ndarray has one column for each training example and 1 row.

    :param fname: Name of file holding the input and target data for training.
    :type fname: string
    :param delim: Delimiter between fields in the text file
    :type delim: one character string
    :return x: ndarray of input vectors
    :rtype x: ndarray of floats of shape (2, m)
    :rtype labels: ndarray of ints of shape (1, m)
    """
    dat1 = np.loadtxt(fname, delimiter=delim)
    m = dat1.shape[0]
    assert dat1.shape[1] >= 3

    labels = dat1[:, 0].reshape(m, 1)
    labels = labels.astype(np.int)
    xs = dat1[:, 1:]

    # We'll also need the positives and the negatives split out eventually,
    # so let's do that now.

    xneg = dat1[dat1[:, 0] < 0.01][:, 1:]
    xpos = dat1[dat1[:, 0] > 0.01][:, 1:]

    return xs, labels, xneg, xpos


def write_model_info(filename, weights, bias, train_accuracy=0.0, test_accuracy=0.0):
    w = weights.ravel().tolist()
    bias0 = bias.item(0)
    slope = -(weights[0, 0] / weights[1, 0])
    y_intercept = -(bias0 / weights[1, 0])

    out_dict = {}
    out_dict['b'] = float(bias0)
    out_dict['slope'] = float(slope)
    out_dict['intercept'] = float(y_intercept)
    out_dict['train_accuracy'] = float(train_accuracy)
    out_dict['test_accuracy'] = float(test_accuracy)
    out_dict['weights'] = w

    with open(filename, 'w') as json_file:
        json.dump(out_dict, json_file, indent=4, sort_keys=True)


def convert_to_one_hot(labels, n_classes):
    res = np.eye(n_classes)[labels.reshape(-1)]
    return res


if __name__ == '__main__':
    ################################################################################
    # Arg parsing
    #

    parser = argparse.ArgumentParser(description='Computer best fit line on a set of points using TensorFlow')

    # Required command-line args
    parser.add_argument('trainpointsfile', help='File containing comma-separated list of x y points for model training')

    # Optional command-line args
    parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Max number of epochs')
    parser.add_argument('-l', '--learnrate', type=float, default=0.01,
                    help='Learning rate')
    parser.add_argument('-f', '--outputfile', default='_tf_logisticregression.png')
    parser.add_argument('-t', '--testpointsfile', default='none')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    args = parser.parse_args()
    train_points_filename = args.trainpointsfile
    test_points_filename = args.testpointsfile
    plot_filename = args.outputfile
    nepochs = args.epochs
    learnrate = args.learnrate
    verbosity = args.debugLevel
    Verbosity = verbosity

    b_out = 0.0

    n_output = 2

    # Read points
    xs, ys_arr, xneg, xpos = read_points_file(train_points_filename)
    n_input = xs.shape[1]

    ys = convert_to_one_hot(ys_arr, n_output)

    print('xs shape: {}'.format(xs.shape))
    print('ys shape: {}'.format(ys.shape))

    if test_points_filename == 'none':
        test_points_filename = train_points_filename

    x_test, y_test_arr, xneg_test, yneg_test = read_points_file(test_points_filename)
    y_test = convert_to_one_hot(y_test_arr, n_output)

    # Define TensorFlow graph for linear regression (expecting 2D points as input).

    # Setup placeholder for input and correct output.
    x = tf.placeholder(tf.float32, [None, n_input])
    y_ = tf.placeholder(tf.float32, [None, n_output])

    # Setup variables for weights as biases.
    W = tf.get_variable('W', shape=(n_input, n_output), initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable('b', initializer=tf.zeros([n_output]))

    # Implement the model.
    logits = tf.matmul(x, W) + b
    # y = tf.nn.sigmoid(logits)
    y = tf.nn.softmax(logits)

    # Compute cross-entropy err:
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    # Graph node for training:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnrate)
    train_step = optimizer.minimize(cost_op)

    # Evaluate accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize all variables.  Do this first in the upcoming session.
    init = tf.initializers.global_variables()

    # Launch the TensorFlow graph session:
    sess = tf.Session()
    sess.run(init)

    # How often to print training stats.
    training_stats_interval = 100
    if nepochs < 100:
        training_stats_interval = 10
    elif nepochs > 80000:
        training_stats_interval = 1000

    if nepochs < 20:
        training_stats_interval = 1

    # Do the training:
    avg_epoch_time = 0.0
    for train_iter in range(nepochs):
        batch_xs = xs
        batch_ys = ys
        t0 = time.time()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        t1 = time.time()
        td = t1 - t0
        avg_epoch_time += td
        if train_iter % training_stats_interval == 0:
            y_out, y_lbl, loss, train_accuracy = sess.run([y, y_, cost_op, accuracy],
                                                           feed_dict={x: batch_xs, y_: batch_ys})
            test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print("Epoch: {}\tloss: {:.4f}\tTrain Accuracy: {:.4f}\tTest Accuracy: {:.4f}".format(train_iter, loss, train_accuracy, test_accuracy))

    # Accuracy:
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    print("Final test accuracy:\t{:.4f}".format(test_accuracy))

    # Average train time per epoch:
    avg_epoch_time /= nepochs
    print('Average train time per epoch (secs): {:.6f}'.format(avg_epoch_time))

    print("Done")

