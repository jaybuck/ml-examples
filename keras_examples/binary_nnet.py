#!/usr/bin/env python
"""Simple neural net for binary classification"""

import sys
import os
import time
import argparse

import numpy as np
from sklearn import utils

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import max_norm

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
    parser = argparse.ArgumentParser(description='Simple neural net model for binary classification using TensorFlow')

    # Optional command-line args
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Max number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=64,
                        help='Mini-batch size')
    parser.add_argument('-r', '--dropout', type=float, default=0.2,
                        help='Dropout probability for dense layers')
    parser.add_argument('-l', '--learnrate', type=float, default=0.0001,
                        help='Learning rate')

    parser.add_argument('--valfraction', type=float, default=0.1,
                        help='Fraction of labeled dataset to use as validation set')
    parser.add_argument('-d', '--datadir', default='data',
                        help='Directory holding data (absolute)')
    parser.add_argument('-s', '--summariesdir', default='mnist_cnn_keras_logs',
                        help='TensorBoard Summaries directory')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')

    args = parser.parse_args()
    nepochs = args.epochs
    batch_size = args.batchsize
    learnrate = args.learnrate
    dropoutrate = args.dropout
    val_fraction = args.valfraction

    homedir = os.path.expanduser('~')
    data_dir = args.datadir
    summaries_dir = os.path.join(homedir, args.summariesdir)

    num_classes = 1

    # Input image dimensions:
    nrows = 28
    ncols = 64

    # Jahir's directory holding Akita data
    # including his massaged numpy datasets holding embeddings of DNA sequences.
    # and the ACE gate this strain fell into.
    data_dir = "/nfs/ddb/strainsim/jmgutierrez"

    # Load numpy data. Then split into train and test sets.
    x_filename = os.path.join(data_dir, "X_PL2665_P4_clean.npy")
    y_filename = os.path.join(data_dir, "y_PL2665_P4_clean.npy")

    x_initial = np.load(x_filename)
    y_initial = np.load(y_filename)

    print('x_initial shape:', x_initial.shape)
    print('y_initial shape:', y_initial.shape)

    nx = x_initial.shape[0]
    ny = y_initial.shape[0]
    assert nx == ny, "Number of rows for X and y are not equal"

    nx_cols = x_initial.shape[1]
    print("X shape:", x_initial.shape)

    # Shuffle X and y arrays in unison before splitting into train and validation
 sets.
    x_all, y_all = utils.shuffle(x_initial, y_initial)

    # Split into train and validation sets.
    val_split_indx = int((1.0 - val_fraction) * nx)
    print('val_split_indx', val_split_indx)
    x_train = x_all[0:val_split_indx]
    x_val = x_all[val_split_indx:]
    y_train = y_all[0: val_split_indx]
    y_val = y_all[val_split_indx:]

    print('x_train shape: {}\tdtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train shape: {}\tdtype {}'.format(y_train.shape, y_train.dtype))

    n_train = x_train.shape[0]
    n_test = x_val.shape[0]
    input_shape = nx_cols

    # Convert to the correct type for input to Keras Dense layer
    # x_train = x_train.astype('float32')
    # x_val = x_val.astype('float32')

    # Should we normalize X here?
    # Right now we expect the input X to be normalized.

    print('\nAfter setting up train and val sets:')
    print('x_train shape', x_train.shape)
    print('# training samples:', n_train)
    print('y_train shape:', y_train.shape)
    print('y_train head:', y_train[0: 10])

    print('\nx_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape)
    print('y_val head:', y_val[0: 10])

    # Convert targets to one-hot vectors:
    # Not needed for this binary classification task.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_val = keras.utils.to_categorical(y_val, num_classes)
    # print('y_test one-hot:\n', y_val[0:10])

    # sys.exit(0)

    # Define neural network model for binary classifier:
    model = keras.models.Sequential()

    # First fully connected layer
    # Using dropout in each of the hidden layers
    model.add(keras.layers.Dropout(dropoutrate))
    model.add(keras.layers.Dense(24, input_dim=nx_cols, activation='relu', kernel
_constraint=max_norm(3)))

    # Second fully connected layer
    model.add(keras.layers.Dropout(dropoutrate))
    model.add(keras.layers.Dense(8, activation='relu', kernel_constraint=max_norm
(3)))

    # Final layer
    model.add(keras.layers.Dropout(dropoutrate))
    model.add(keras.layers.Dense(1, activation='sigmoid', kernel_constraint=max_n
orm(3)))

    # Compile the model:
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learnrate),
                  metrics=['accuracy'])

    # Setup logging:
    history_callback = AccuracyHistory()

    # Train the model:
    t0 = time.time()
    trainhist = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=nepochs,
                        verbose=1,
                        validation_data=(x_val, y_val),
                        callbacks=[history_callback])

    tdelta = time.time() - t0
    print(f"Training time (sec): {tdelta:8.3f}")

    # Score the model:
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # history keys:  dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
    print("history keys:", trainhist.history.keys())
    print("history loss train:", trainhist.history['loss'])
    print("history loss val:", trainhist.history['val_loss'])
    print("history accuracy train:", trainhist.history['accuracy'])
    print("history accuracy val:", trainhist.history['val_accuracy'])

    plt.figure()
    # plt.plot(range(1, nepochs + 1), history_callback.acc)
    plt.plot(trainhist.history['accuracy'], label='train')
    plt.plot(trainhist.history['val_accuracy'], label='val')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Model train and val accuracy')
    plt.savefig('_test_accuracy.png')
    plt.close()

    print('Done')



