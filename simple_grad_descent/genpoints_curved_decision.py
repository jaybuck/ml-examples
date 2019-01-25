#!/usr/bin/env python
"""
Generate a set of x, y points in 2D with labels such that:
label = 1 if the y coordinate is > -cos(x) (with a little noise tossed in)
        0 otherwise
"""

import numpy as np
import matplotlib.pyplot as plt

xmax = 3.0
xmin = -xmax
ymax = 2.0
ymin = -ymax

npoints = 2000
noise_scale = 0.1

x_coord = np.random.uniform(xmin, xmax, size=npoints)
y_coord = np.random.uniform(ymin, ymax, size=npoints)
labels = np.array([int(y > -np.cos(x) + np.random.normal(scale=noise_scale)) for x, y in zip(x_coord, y_coord)])

with open('_points_cosx.txt', 'w') as ofile:
    for i in range(len(x_coord)):
        ofile.write('{0},{1},{2}\n'.format(labels[i], x_coord[i], y_coord[i]))

# Plot the points
colors = ['red', 'blue']
plt.axis([xmin, xmax, ymin, ymax])
plt.axis('equal')
plt.grid()
plt.scatter(x_coord, y_coord, s=4, marker='.', color=[colors[l] for l in labels] )
plt.savefig('_points_cosx.png')

