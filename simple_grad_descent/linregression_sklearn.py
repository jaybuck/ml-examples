#!/usr/bin/env python

# Run scikit-learn linear regression on a set of x y points.

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

################################################################################
# Arg parsing
#

parser = argparse.ArgumentParser(description='Run sklearn linear_model functions on a set of points')

# Required command-line args
parser.add_argument('pointsfile', help='File containing tab-separated list of x y points')

# Optional command-line args
parser.add_argument('-f', '--outputfile', default='_regression_sklearn.png')

args = parser.parse_args()
points_filename = args.pointsfile
plot_filename = args.outputfile

################################################################################
# Read points
#
# Each line of the points file is expected to be tab-delimited with format:
# x y
# where x, y are floats

dat1 = np.loadtxt(points_filename, delimiter='\t')

# Create arrays of x any  values.
# The reshape at the end of the slice lines changes the dimension of the array
# from being a 1 dimensional row vector:
# [0.0, 0.5, 1.0, ... , 10.0]
# with numpy shape (20,)
# to a 2 dimensional array with 20 rows and 1 column:
# [ [0.0],
#   [0.5],
#   [1.0],
# ...
#   [10.0]]
# with numpy shape: (20,1)
# That's what the LinearRegression.fit() function wanted.

xs = dat1[:, 0].reshape(-1, 1)
ys = dat1[:, 1].reshape(-1, 1)

# Compute the best-fit line for this point cloud.
reg_model = linear_model.LinearRegression()
reg_model.fit(xs, ys)
r2_score = reg_model.score(xs, ys)

# print('The best-fit model attributes')
# print('coef_ {0}'.format(reg_model.coef_))
# print('intercept_ {0}'.format(reg_model.intercept_))

# Print out the slope and intercept:
# NB: For this linear model, the coeff_ array only has one element, the zero-th element.
slope_fit = reg_model.coef_[0][0]
intercept_fit = reg_model.intercept_[0]
print('Best-fit Slope: {0}\tIntercept: {1}'.format(slope_fit, intercept_fit))
print('R2 score: {}'.format(r2_score))

# Plot the points and the best-fit line

# Build two points to plot the best-fit line
xmin = xs.min()
xmax = xs.max()

x_fit = [xmin, xmax]
y_fit = [slope_fit * x + intercept_fit for x in x_fit]

# Or:

x_fit = np.array([xmin, xmax])
y_fit = slope_fit * x_fit + intercept_fit

# Plot the points and line:
plt.figure()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.axis('equal')
plt.plot(xs, ys, 'bo', label='point cloud')
plt.plot(x_fit, y_fit, 'r-', label='best fit line')
plt.legend()
plt.savefig(plot_filename)
