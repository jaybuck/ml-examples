#!/usr/bin/env python
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

npoints = 50
slope = 0.5
b = 2.0

x = np.linspace(0.0, 10.0, npoints, endpoint=False)
urand = np.random.uniform(-1.0, 1.0, npoints)
y = slope * x + b + urand

with open('_points.txt', 'w') as ofile:
    for i in range(len(x)):
        ofile.write('{0}\t{1}\n'.format(x[i], y[i]))

plt.plot(x, y, 'x')
plt.axis('equal')
plt.axis([-1, 10, -1, 10])
plt.grid()
plt.savefig('_pointcloud.png')
