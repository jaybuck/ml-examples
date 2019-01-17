#!/usr/bin/env python

# Compute best fit line on a set of x y points using gradient descent
# naively.

import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_points_file(fname, delim='\t'):
    """Read x y points from a text file, returning numpy arrays containing the x coordinates and the y coordinates"""
    dat1 = np.loadtxt(fname, delimiter=delim)
    xs = dat1[:, 0]
    ys = dat1[:, 1]
    return xs, ys


class LinearRegressionModelNaive:
    """
    Class used to implement linear regression model, naively.

    Attributes 
    ----------
    coef_ : linear model coefficients
    intercept_ : linear model intercept

    Methods
    -------
    fit(x, y, nepochs) : compute coefficients of best fit line
    """

    def __init__(self, slope=0.0, intercept=0.0):
        """
        Parameters
        ----------
        slope : float, optional
            Initial value of slope
        intercept : float, optional
            Initial value of intercept
        """

        self.coef_ = np.zeros(1)
        self.coef_[0] = slope
        self.intercept_ = intercept
        self.cost_ = 0.0
        self.r2_score = -999999999.0

    def fit(self, xs, ys, nepochs=10, learnrate=0.01):
        """Computes slope and intercept of best fit line for points.

        Parameters
        ----------
        xs : numpy array of shape (m,) floats where m is the number of points
            The x coordinates of the points
        ys : numpy array of shape (m,) floats
            The y coordinates of the points
        nepochs : int
            Number of training epochs
        learnrate : float
            Learning rate

        Action 
        Computes the slope and interecept of the best-fit line for a set of points
        and put those into coef_[0][0] and intercept_ , respectively.
        """

        # A couple checks on the parameters:

        # The xs and ys must be the same length
        assert(xs.shape == ys.shape)
        assert(nepochs > 0)

        w = self.coef_[0]
        b = self.intercept_

        # We'll need the number of points a lot:
        m = xs.shape[0]

        # The mean of the y values will be used for evaluating the model.
        y_mean = ys.mean()

        sys.stderr.write('m: {}\n'.format(m))
        sys.stderr.write('y_mean: {}\n'.format(y_mean))
        sys.stderr.write('Initial w {}    b {}\n'.format(w, b))

        t0 = time.perf_counter()

        for epoch in range(nepochs):
            # Compute cost J and gradients at each epoch
            J = 0.0
            dw = 0.0
            db = 0.0

            for i in range(m):
                # Compute activation (or yhat) for x_i using current w and b.
                a = w * xs[i] + b
                # Use mean squared error loss
                J += (ys[i] - a)**2
                dw += xs[i] * (a - ys[i])
                db += (a - ys[i])

            J /= m
            dw /= m
            db /= m

            w -= learnrate * dw
            b -= learnrate * db

            if (epoch % 10) == 0:
                sys.stderr.write('epoch: {}\tcost: {}\tw: {}\tb: {}\n'.format(epoch, J, w, b))

        t1 = time.perf_counter()
        sys.stderr.write('Train time: {0:10.4f} sec\n'.format(t1 - t0))

        # Evaluations:
        # Compute R^2 metric
        null_model_v = 0.0
        for i in range(m):
            null_model_v += (ys[i] - y_mean)**2

        null_model_v /= m
        self.r2_score = 1.0 - J / null_model_v

        self.coef_[0] = w
        self.intercept_ = b
        self.cost_ = J
        return w, b


if __name__ == '__main__':
    ################################################################################
    # Arg parsing
    #

    parser = argparse.ArgumentParser(description='Computer best fit line on a set of points, naively')

    # Required command-line args
    parser.add_argument('-e', '--epochs', type=int, default=10,
                    help='Max number of epochs')
    parser.add_argument('-l', '--learnrate', type=float, default=0.01,
                    help='Learning rate')
    parser.add_argument('pointsfile', help='File containing tab-separated list of x y points')

    # Optional command-line args
    parser.add_argument('-f', '--outputfile', default='_regression_naive.png')

    args = parser.parse_args()
    points_filename = args.pointsfile
    plot_filename = args.outputfile
    nepochs = args.epochs
    learnrate = args.learnrate

    # Read points
    xs, ys = read_points_file(points_filename)


    ################################################################################
    # Compute best fit line coefficients
    #
    linregmodel = LinearRegressionModelNaive(slope=0.6, intercept=2.0)
    slope, b = linregmodel.fit(xs, ys, nepochs=nepochs, learnrate=learnrate)
    print('Best fit model: cost {}\tslope: {}\tb: {}'.format(linregmodel.cost_, slope, b))
    print('R^2 score: {}'.format(linregmodel.r2_score))

    # Plot the points and the best-fit line

    # Build two points to plot the best-fit line
    xmin = xs.min()
    xmax = xs.max()

    x_fit = np.array([xmin, xmax])
    y_fit = slope * x_fit + b

    # Plot the points and line:
    plt.figure()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Linear Regression Example (not vectorized)')
    plt.axis('equal')
    plt.plot(xs, ys, 'bo', label='point cloud')
    plt.plot(x_fit, y_fit, 'r-', label='best fit line')
    plt.legend()
    plt.savefig(plot_filename)
