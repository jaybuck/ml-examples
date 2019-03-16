# simple_grad_descent
Examples of simple code to implement two machine learning algorithms: linear regression and 
logistic regression.

Here the tasks take as input 2D points. 
* For linear regression, the goal of the training is to find the line that best fits
the data. That is, find the two parameters, slope and intercept, that define that line.
  * The cost function is mean squared error. The best line is the one which minimizes this cost.
* For logistic regression we have two blobs of points that overlap and the task for
the training algorithm is to find the line that does the best job of separating
the two blobs. Since the blobs overlap it will never achieve zero error.
  * The cost function is cross-entropy. Again, we define the best line as the one that
  minimizes this cost.
  
The programs:
* `linregression2_naive.py` : Implements linear regression using gradient descent on the cost, 
which is mean-squared error. The implementation is naive because it uses for loops to
compute the cost instead of vectorized functions.
* `linregression2.py` : Implements linear regression using gradient descent on the cost, 
which is mean-squared error. The implementation uses vectorized numpy functions to
compute the cost.
* `logisticregression.py` : Implements logistic regression. This code takes as inputs 
2D points, each with a class label. 
The classes in this task are in {0, 1}.
The training algorithm modifies the parameters of the logistic regression model
to reduce the cross-entropy cost function.

In addition to stand-alone programs, this directory contains two Jupyter notebooks 
that provide an easy way to see these two algorithms in action:

* `linregression2_naive.ipynb` : Implements linear regression as above.
* `linregression2.ipynb` : Implements linear regression as above.
* `logisticregression_np.ipynb` : Implements logistic regression as above.
