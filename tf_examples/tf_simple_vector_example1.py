#!/usr/bin/env python
"""
Simple examples of vector arithmetic in TensorFlow.
"""

import tensorflow as tf
import numpy as np

# Setup interactive TensorFlow session:
sess = tf.InteractiveSession()
print('TensorFlow vector arithmetic examples\n')
print("First, let's create a couple scalars:")
a = tf.constant(2.0)
b = tf.constant(3.0)

(a_out, b_out) = sess.run([a, b])

print('a: {}\tb: {}\n'.format(a_out, b_out))

print('Create a couple vectors:')
v0 = tf.constant([0.0, 1.0, 2.0, 3.0], dtype=tf.float32, name='v0')
v1 = tf.constant([1.1, 2.2, 3.3, 4.4], dtype=tf.float32, name='v1')

v0_out, v0_shape = sess.run([v0, tf.shape(v0)])
v1_out, v1_shape = sess.run([v1, tf.shape(v1)])

print('v0: {}\tshape: {}'.format(v0_out, v0_shape))
print('v1: {}\tshape: {}\n'.format(v1_out, v1_shape))

print('Do some simple scalar vector arithmetic:')
print('v0 + a: {}'.format(sess.run(v0 + a)))
print('v0 * b: {}\n'.format(sess.run(v0 * b)))

v01_sum = v0 + v1
v01_prod = v0 * v1


print('Do some simple vector arithmetic:')
print('v0 + v1 : {}'.format(sess.run(v01_sum)))
print('v0 * v1 : {}\n'.format(sess.run(v01_prod)))


print('There are several ways to take the dot product of two vectors')
print('The most straightforward to to use tf.tensordot()')

v01_dot = tf.tensordot(v0, v1, 1)

print('v0 tensordot v1: {}\n'.format(sess.run(v01_dot)))

print('You can also use tf.multiply() followed by tf.reduce_sum() :')

v01_reduce = tf.reduce_sum(v01_prod)
print('v0 dot v1 using tf.reduce_sum(): {}\n'.format(sess.run(v01_reduce)))

print('\nNow for a couple matrix examples:')

print("Let's create a couple matrices.")
print("First by just reshaping the 1D vectors we used above into square matrices:")
mat0 = tf.reshape(v0, [2, 2])
mat1 = tf.reshape(v1, [2, 2])

m0_out, m0_shape = sess.run([mat0, tf.shape(mat0)])
print('mat0: shape: {}\n{}\n'.format(m0_shape, m0_out))

m1_out, m1_shape = sess.run([mat1, tf.shape(mat1)])
print('mat1: shape: {}\n{}\n'.format(m1_shape, m1_out))

print('Then create one more matrix by concatenating a row to one of these.')
print("We'll use this for another matrix multiply.")
mat3 = tf.concat([mat0, [[4.0, 5.0]]], 0)
m3_shape, m3 = sess.run([tf.shape(mat3), mat3])
print('mat3: shape: {}\n{}\n'.format(m3_shape, m3))

m01_sum = mat0 + mat1
print('m0 + m1: \n{}\n'.format(sess.run(m01_sum)))

m01_matmul = tf.matmul(mat0, mat1)
print('m0 matmul m1: \n{}\n'.format(sess.run(m01_matmul)))

m30_matmul = tf.matmul(mat3, mat0)
print('m3 matmul m0: \n{}'.format(sess.run(m30_matmul)))

print('Done')
