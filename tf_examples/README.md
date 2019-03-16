# tf-examples
TensorFlow examples.

This directory contains several examples of TensorFlow programs for both classification and 
showing TensorFlow concepts:

* `tf_arithmetic_example1.py` : Simple example of how to define a TensorFlow graph for
a simple arithmetic computation.
* `tf_arithmetic_example1_tb.py` : Implements the same simple example and adds annotations
for Tensorboard to display the computation graph.
* `tf_logisticregression_naive.py` : Implements a logistic regression model for the 2D point
classification task described in the simple_grad_descent directory. The code for the 
function is naive in that it is easy to understand, but not numerically stable. 
* `tf_logisticregression.py` : Implements a logistic regression model for the 2D point
classification task described in the simple_grad_descent directory in the way that works
best in TensorFlow. Basically, instead of "hand coding" the cross-validation cost function
we use one of TensorFlow's built-in functions to do that.
