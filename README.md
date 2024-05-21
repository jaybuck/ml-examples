# ml-examples
Examples of code to implement simple machine learning algorithms.

ToDo:
* Setup a new venv for these
* Update these, esp variable naming in regression code
  * Check the algorithms, esp the number of output units
* Add PyTorch examples?

The code in this repo is intended as part of a series of introductory tutorials on machine learning. The key topic areas are:
* A brief introductory Python tutorial.
* Understanding gradient descent. These examples explicitly code 
gradient descent in Python Numpy to show how gradient descent works. 
These are in the [simple_grad_descent](simple_grad_descent/README.md) directory.
Initial examples are:
  * Linear regression
  * Logistic regression
* Using TensorFlow to implement machine learning models.
  * We start by presenting a couple very simple examples where we do 
  arithmetic on scalars, vectors, and matrices.
  These are in the [tf_examples](tf_examples/README.md) directory.
  * We then take the logistic regression model in the previous section and 
  recode it using TensorFlow. TensorFlow of course takes care of modifying model weights 
  using gradient descent   which makes life much easier for the researcher. 
  Of course it also hides how it's done. 
  And one of the goals which was the goals of the the exercises in the previous section
  was to explicitly show how gradient descent can be computed so one will understand
  it once tools like TensorFlow are doing that work for us.
  
  With this foundation, we move to a more interesting task: classifying handwritten
  digits. This is a highly non-linear task and linear models such as
  logistic regression will not be able to achieve high accuracy on it.
  So we define and run a convolutional neural network that exploits the structure
  of the images to learn model parameters that enable the model to achieve 
  high accuracy. The code for this is in the [mnist_oldstyle](mnist_oldstyle/README.md) 
  directory.

These exercises assume you have Python 3 setup and these packages installed:
* numpy
* pandas
* sklearn
* matplotlib
* seaborn
* tensorflow (used version 1.12 for these examples)

We used Anaconda with Python 3.6 for these examples. (Because TensorFlow does not yet work with Python 3.7)
