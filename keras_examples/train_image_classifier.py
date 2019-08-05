#!/usr/bin/env python
"""Train CNN to classify geological images"""

import sys
import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cv2
import matplotlib.pyplot as plt

import keras
from keras import backend as K


def read_labeled_images(imagelist_filename, datadir=None):
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
    path_list = []
    indx = 0
    for fname in imagenames_pd['filename']:
        # print('\nimagenames_pd filename:', fname)
        pathname = os.path.join(datadir, fname) if datadir is not None else fname
        path_list.append(pathname)
        if indx % 100 == 0:
            print(f'\n{indx}\timage pathname: {pathname}')
        img = cv2.imread(pathname)
        # print('    img shape:', img.shape)
        image_list.append(img)
        indx += 1

    images = np.array(image_list)
    print('images shape:', images.shape)

    # Write the image paths.
    path_pd = pd.DataFrame.from_dict({'label': labels, 'path': path_list})
    path_pd.to_csv('_imagepaths.csv', index=False)

    return images, labels, path_list


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
    parser = argparse.ArgumentParser(description='CNN model for image data using TensorFlow')
    # Required command-line args
    parser.add_argument('trainfile', help='File containing list of training images with labels')
    # parser.add_argument('testfile', help='File containing list of test images with labels')

    # Optional command-line args
    parser.add_argument('-c', '--classes', type=int, default=6,
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
    parser.add_argument('-s', '--summariesdir', default='image_cnn_logs',
                        help='TensorBoard Summaries directory')

    args = parser.parse_args()
    train_filename = args.trainfile
    # test_filename = args.testfile

    num_classes = args.classes
    nepochs = args.epochs
    batch_size = args.batchsize
    learnrate = args.learnrate
    dropoutrate = args.dropout

    n_nexttolast = 64

    homedir = os.path.expanduser('~')
    data_dir = os.path.join(homedir, args.datadir)
    summaries_dir = os.path.join(homedir, args.summariesdir)

    # image_dir = os.path.join(homedir, 'awork/ImageWork/imageExp1/people')
    # image_path = os.path.join(image_dir, '09.jpg')
    #
    # img = cv2.imread(image_path)
    # print('img shape: {}\tdtype: {}'.format(img.shape, img.dtype))

    # Read train and test images and labels:
    images, y_all, image_paths = read_labeled_images(train_filename, datadir=data_dir)
    x_all = images.astype('float32') / 255
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.1)

    x_shape = x_train.shape
    n_train = x_shape[0]
    nrows = x_shape[1]
    ncols = x_shape[2]
    n_channels = x_shape[3]

    # x_test, y_test = read_labeled_images(test_filename, datadir=data_dir)
    n_test = x_test.shape[0]

    # Reshape the image data into 4D tensor: (sample_number, x_img_size, y_img_size, num_channels)
    # x_train = x_train.reshape(n_train, nrows, ncols, n_channels)
    # x_test = x_test.reshape(n_test, nrows, ncols, n_channels)
    input_shape = (nrows, ncols, n_channels)

    # Convert to the correct type for input to CNN:
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
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
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3),
                                  activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Third conv layer
    # model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),
    #                               activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropoutrate))

    model.add(keras.layers.Dense(1024, activation='sigmoid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropoutrate))

    model.add(keras.layers.Dense(n_nexttolast, activation='sigmoid'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(dropoutrate))

    # Final layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model:
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learnrate),
                  metrics=['accuracy'])

    # Setup logging:
    history_callback = AccuracyHistory()

    # Show model architecture
    model.summary()

    # Train the model:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=nepochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[history_callback])

    # Save the model architecture
    # model_json = model.to_json()
    # with open('_model.json', 'w') as json_file:
    #     json_file.write(model_json)

    # Save weights
    # model.save_weights('_model.h5')

    # Save model architecture and weights
    model.save('_model.h5')
    print('Model saved')

    # Score the model:
    score = model.evaluate(x_test, y_test, verbose=1)
    print('score:', score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    plt.plot(history_callback.indices, history_callback.acc, label='train')
    plt.plot(history_callback.indices, history_callback.val_acc, label='test')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.savefig('_traintest_accuracy.png')
    plt.close()

    # Get penultimate fully-connected layer
    nexttolast_layer = model.get_layer('dense_2')
    get_nexttolast_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                             [nexttolast_layer.output])
    nexttolast_output = get_nexttolast_layer_output([x_all, 0])[0]
    print('nexttolast_output shape ', nexttolast_output.shape)
    np.save('_nexttolast_a.npy', nexttolast_output)

    print('Done')
