# ml-examples
Examples of code to implement simple machine learning algorithms.

The code in this repo is intended as part of a series of introductory tutorials on machine learning. The key topic areas are:
* Understanding gradient descent. These examples explicitly code gradient descent in Python Numpy to show how it works. Initial examples are:
  * Linear regression
  * Logistic regression
* Using TensorFlow to implement machine learning models.
  * We start by taking the logistic regression model in the previous section and recode it using TensorFlow. TensorFlow of course takes care of modifying model wieghts using gradient descent which makes life much easier for the researcher. Of course it also hides how it's done, which was the goal of the exercises in the previous section.

These exercises assume you have Python 3 setup and these modules loaded:
* numpy
* pandas
* sklearn
* matplotlib
* seaborn
* tensorflow (used 1.09 for these examples)
