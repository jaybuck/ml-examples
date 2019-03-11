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
    y_intercept = -(bias0 / w1[1, 0])

    out_dict = {}
    out_dict['b'] = float(bias0)
    out_dict['slope'] = float(slope)
    out_dict['intercept'] = float(y_intercept)
    out_dict['train_accuracy'] = float(train_accuracy)
    out_dict['test_accuracy'] = float(test_accuracy)
    out_dict['weights'] = w

    with open(filename, 'w') as json_file:
        json.dump(out_dict, json_file, indent=4, sort_keys=True)



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

    # Read points
    xs, ys, xneg, xpos = read_points_file(train_points_filename)

    print('xs shape: {}'.format(xs.shape))
    print('ys shape: {}'.format(ys.shape))

    if test_points_filename == 'none':
        test_points_filename = train_points_filename

    x_test, y_test, xneg_test, yneg_test = read_points_file(test_points_filename)

    # Define TensorFlow graph for linear regression (expecting 2D points as input).

    # Setup placeholder for input and correct output.
    x = tf.placeholder(tf.float32, [None, 2])
    y_ = tf.placeholder(tf.float32, [None, 1])

    # Setup variables for weights as biases.
    W = tf.get_variable('W', shape=(2, 1), initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable('b', initializer=tf.zeros([1]))

    # Implement the model.
    logits = tf.matmul(x, W) + b
    y = tf.nn.sigmoid(logits)

    # Compute cross-entropy err:
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = -(tf.reduce_sum((y_ * tf.log(y)) + ((1.0 - y_) * tf.log(1.0 - y))))

    # Graph node for training:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnrate)
    train_step = optimizer.minimize(cross_entropy)

    # Evaluate accuracy
    accuracy_thresh = tf.constant(0.2)
    correct_prediction = tf.less(tf.abs(y_ - y), accuracy_thresh)
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
            w1, b_out, y_out, y_lbl, loss, accuracy1 = sess.run([W, b, y, y_, cross_entropy, accuracy],
                                                                feed_dict={x: batch_xs, y_: batch_ys})
            test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
            print("Epoch: {}\tloss: {:.4f}\tTrain Accuracy: {:.4f}\tTest Accuracy: {:.4f}".format(train_iter, loss, accuracy1, test_accuracy))
            # print("b: ", b_out)
            # print("W: ", w1)
            # print("y: ", y_out[0:10, :], "\nlbl: ", y_lbl[0:10, :])

    # Accuracy:
    test_accuracy = sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    print("Final test accuracy:\t{:.4f}".format(test_accuracy))

    # Average train time per epoch:
    avg_epoch_time /= nepochs
    print('Average train time per epoch (secs): {:.6f}'.format(avg_epoch_time))

    write_model_info('_tf_logressionmodel_test.json', w1, b_out, train_accuracy=accuracy1, test_accuracy=test_accuracy)

    # Plot the training points and the best-fit line

    # Using w and b, compute the decision surface for the logistic regression model.
    # Which is a line s.t.
    # w0 * x + w1 * x1 + b = 0.5
    # or, to compute the slope and intercept of that line:
    # x1 = -(w0/x1)*x0 (-b + 0.5)/w1

    # Build two points to plot the best-fit line
    xmin = xs[:, 0].min()
    xmax = xs[:, 0].max()

    b_val = b_out.item(0)
    slope = -(w1[0, 0] / w1[1, 0])
    y_intercept = -(b_val / w1[1, 0])
    print('Decision line slope: {:.2f}\tintercept: {:.2f}'.format(slope, y_intercept))
    x_fit = np.array([xmin, xmax])
    y_fit = slope * x_fit + y_intercept

    # Get the x, y coordinates of the points for plotting:
    x0_pos = xpos[:, 0]
    x1_pos = xpos[:, 1]
    x0_neg = xneg[:, 0]
    x1_neg = xneg[:, 1]

    # Plot the points and decision surface line:
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('TensorFlow Logistic Regression Example: Training Data')
    plt.axis('equal')
    plt.scatter(x0_neg, x1_neg, s=4, c='r', marker='.', alpha=0.5, label='negatives')
    plt.scatter(x0_pos, x1_pos, s=4, c='b', marker='.', alpha=0.5, label='positives')
    plt.plot(x_fit, y_fit, linestyle='-', color='deeppink', label='model decision surface')
    plt.legend()
    plt.savefig('_train' + plot_filename)
    plt.close()

    print("Done")

