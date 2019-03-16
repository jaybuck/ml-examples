# mnist-oldstyle
Examples of TensorFlow code to implement classifiers for the MNIST digit classification task.

The example programs implement classification using logistic regression and
convolutional neural networks (ConvNets).
Using TensorFlow to implement these machine learning models we have:
* `mnist1.py` : Implements a logistic regression classifier for this task.
* `mnist_cnn.py` : Implements a convolutional neural network classifier for this task.
  * The model implemented in this program is relatively simple, with two convolution layers
  and two fully-connected layers.
  * Each convolution layer uses the reLU activation function and is followed by a max-pooling layer.
  * The fully-connected layers use dropout.

This directory is called oldstyle because it still uses `feed-dict` to input the training examples
into the model, instead of the newer TensorFlow `Dataset`.
