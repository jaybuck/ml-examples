#!/usr/bin/env python
import sys

import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:]
f_args = [float(x) for x in args]

lbl = 0
mean = np.array([0, 4])
print('mean: {}'.format(mean))

cov = np.array([[2.0, .2], [.2, .2]])
if len(args) >= 4:
    cov = np.array([[f_args[0], f_args[1]], [f_args[2], f_args[3]]])

print('cov: \n{}'.format(cov))

npoints = 1000

x, y = np.random.multivariate_normal(mean, cov, npoints).T

with open('_points.txt', 'w') as ofile:
    for i in range(len(x)):
        ofile.write('{0},{1},{2}\n'.format(lbl, x[i], y[i]))

# Plot the points
plt.axis([-10, 10, -10, 10])
plt.axis('equal')
plt.grid()
plt.plot(x, y, 'x')
plt.savefig('_pointcloud.png')
