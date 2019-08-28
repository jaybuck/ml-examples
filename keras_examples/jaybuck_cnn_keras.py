#!/usr/bin/env python
"""Keras CNN for jaybuck images"""

import sys
import os
import time
import argparse

import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras import metrics

import cv2

import matplotlib.pyplot as plt

def read_labeled_images(imagelist_filename):
    print('read_labeled_images')
    imagenames_pd = pd.read_csv(imagelist_filename, header=None, names=['label', 'filename'])
    print('label column dtype:', imagenames_pd['label'].dtype)
    print('filename column dtype:', imagenames_pd['filename'].dtype)
    labels_list = imagenames_pd['label'].tolist()
    labels = np.array(labels_list, dtype=np.int32)
    print('labels shape:', labels.shape)
    print('labels head:', labels[0:10])
    print("==============")

    # Read the images:
    image_list = []
    for fname in imagenames_pd['filename']:
        print('\nimagenames_pd filename:', fname)
        img = cv2.imread(fname)
        print('    img shape:', img.shape)
        image_list.append(img)

    images = np.array(image_list)
    print('images shape:', images.shape)
    return images, labels


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.index = 0
        self.indices = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.indices.append(self.index)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.index += 1


if __name__ == '__main__':

    ################################################################################
    # Arg parsing
    #
    parser = argparse.ArgumentParser(description='CNN model for MNIST data using TensorFlow')
    # Required command-line args
    parser.add_argument('trainfile', help='File containing list of training images with labels')
    parser.add_argument('testfile', help='File containing list of test images with labels')

    # Optional command-line args
    parser.add_argument('-c', '--classes', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Max number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='Mini-batch size')
    parser.add_argument('-r', '--dropout', type=float, default=0.2,
                        help='Keep probability for training dropout')
    parser.add_argument('-l', '--learnrate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    parser.add_argument('-d', '--datadir', default='data',
                        help='Directory holding MNIST data wrt your homedir')
    parser.add_argument('-s', '--summariesdir', default='jaybuck_cnn_keras_logs',
                        help='TensorBoard Summaries directory')

    args = parser.parse_args()
    train_filename = args.trainfile
    test_filename = args.testfile

    num_classes = args.classes
    nepochs = args.epochs
    batch_size = args.batchsize
    learnrate = args.learnrate
    dropoutrate = args.dropout

    homedir = os.path.expanduser('~')
    data_dir = os.path.join(homedir, args.datadir)
    summaries_dir = os.path.join(homedir, args.summariesdir)

    # image_dir = os.path.join(homedir, 'awork/ImageWork/imageExp1/people')
    # image_path = os.path.join(image_dir, '09.jpg')
    #
    # img = cv2.imread(image_path)
    # print('img shape: {}\tdtype: {}'.format(img.shape, img.dtype))

    # Read train and test images and labels:
    x_train, y_train = read_labeled_images(train_filename)
    x_shape = x_train.shape
    n_train = x_shape[0]
    nrows = x_shape[1]
    ncols = x_shape[2]
    n_channels = x_shape[3]

    x_test, y_test = read_labeled_images(test_filename)
    n_test = x_test.shape[0]

    # Reshape the image data into 4D tensor: (sample_number, x_img_size, y_img_size, num_channels)
    # x_train = x_train.reshape(n_train, nrows, ncols, n_channels)
    # x_test = x_test.reshape(n_test, nrows, ncols, n_channels)
    input_shape = (nrows, ncols, n_channels)

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
    # model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
    #                               strides=(1, 1),
    #                               activation='relu',
    #                               input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
                                  strides=(1, 1),
                                  input_shape=input_shape,
                                  kernel_initializer='he_normal',
                                  use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # model.add(keras.layers.Activation('relu'))
    # model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Second conv layer
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
                                  kernel_initializer='he_normal',
                                  use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Third conv layer
    # model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),
    #                               activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(dropoutrate))

    model.add(keras.layers.Dense(1024, use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # Final layer
    model.add(keras.layers.Dropout(dropoutrate))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learnrate),
                  metrics=['accuracy'])

    # Setup logging:
    history_callback = AccuracyHistory()

    # Train the model:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nepochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history_callback])

    # Score the model:
    print('Experiments using batch norm: Dense -> BatchNorm -> reLU\n')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('score:', score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    plt.plot(history_callback.indices, history_callback.acc, label='train')
    plt.plot(history_callback.indices, history_callback.val_acc, label='test')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.savefig('_jaybuck_traintest_accuracy.png')
    plt.close()

    print('Done')

