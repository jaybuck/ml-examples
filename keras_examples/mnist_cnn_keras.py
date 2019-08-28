#!/usr/bin/env python
"""CNN for MNIST in Keras."""

import sys
import os
import time
import argparse

import numpy as np
import keras
from keras.datasets import mnist
from keras import metrics

import matplotlib.pyplot as plt


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == '__main__':

    ################################################################################
    # Arg parsing
    #
    parser = argparse.ArgumentParser(description='CNN model for MNIST data using TensorFlow')

    # Optional command-line args
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Max number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('-r', '--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout')
    parser.add_argument('-l', '--learnrate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    parser.add_argument('-d', '--datadir', default='data',
                        help='Directory holding MNIST data wrt your homedir')
    parser.add_argument('-s', '--summariesdir', default='mnist_cnn_keras_logs',
                        help='TensorBoard Summaries directory')

    args = parser.parse_args()
    nepochs = args.epochs
    batch_size = args.batchsize
    learnrate = args.learnrate
    dropoutrate = args.dropout

    homedir = os.path.expanduser('~')
    data_dir = os.path.join(homedir, args.datadir)
    summaries_dir = os.path.join(homedir, args.summariesdir)

    num_classes = 10

    # Input image dimensions:
    nrows = 28
    ncols = 28

    # Load MNIST data. Already split into train and test sets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print('x_train shape: {}\tdtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train shape: {}\tdtype {}'.format(y_train.shape, y_train.dtype))

    # Reshape the image data into 4D tensor: (sample_number, x_img_size, y_img_size, num_channels)
    # MNIST data is grayscale so only one channel.
    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    x_train = x_train.reshape(n_train, nrows, ncols, 1)
    x_test = x_test.reshape(n_test, nrows, ncols, 1)
    input_shape = (nrows, ncols, 1)

    # Convert to the correct type for input to CNN:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('\nAfter reshaping:')
    print('x_train shape', x_train.shape)
    print('# training samples:', n_train)
    print('y_train shape:', y_train.shape)
    print('y_train head:', y_train[0: 10])
    print('\nx_test shape:', x_test.shape)
    print('# test samples:', x_test.shape[0])
    print('y_test shape:', y_test.shape)
    print('y_test head:', y_test[0: 10])

    # Convert targets to one-hot vectors:
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_test one-hot:\n', y_test[0:10])

    # sys.exit(0)

    # Define neural network model:
    model = keras.models.Sequential()

    # First conv layer
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
                                  strides=(1, 1),
                                  activation='relu',
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Second conv layer
    model.add(keras.layers.Conv2D(64, kernel_size=(5, 5),
                                  activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu'))

    # Final layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learnrate),
                  metrics=['accuracy'])

    # Setup logging:
    history_callback = AccuracyHistory()

    # Train the model:
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=nepochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[history_callback])

    # Score the model:
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # history keys:  dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

    plt.figure()
    # plt.plot(range(1, nepochs + 1), history_callback.acc)
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Model train and test accuracy')
    plt.savefig('_test_accuracy.png')
    plt.close()

    print('Done')







