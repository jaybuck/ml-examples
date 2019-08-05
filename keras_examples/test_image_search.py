#!/usr/bin/env python
"""Train CNN to classify geological images"""

import os
import argparse

import numpy as np
import pandas as pd
from scipy.spatial import distance

import cv2
import matplotlib.pyplot as plt

import keras
from keras import backend as K

def read_labeled_images(imagelist_filename, datadir=None, has_header=False):
    print('read_labeled_images')
    if has_header:
        imagenames_pd = pd.read_csv(imagelist_filename, header=0, names=['label', 'filename'])
    else:
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
    parser.add_argument('testfile', help='File containing list of test images with labels')

    # Optional command-line args
    parser.add_argument('-c', '--classes', type=int, default=6,
                        help='Number of output classes')
    parser.add_argument('-v', '--debugLevel', type=int, default=0,
                        help='Level of debugging output (verbosity).')
    parser.add_argument('-m', '--model', default='_model.h5',
                        help='Name of model file')
    parser.add_argument('-r', '--hiddenreps', default='_nexttolast_a.npy',
                        help='Name of file holding hidden representations of known image dataset')
    parser.add_argument('-d', '--datadir', default='data',
                        help='Directory holding MNIST data wrt your homedir')

    args = parser.parse_args()
    train_filename = args.trainfile
    test_filename = args.testfile

    num_classes = args.classes

    model_filename = args.model
    nexttolast_reps_filename = args.hiddenreps

    homedir = os.path.expanduser('~')
    data_dir = os.path.join(homedir, args.datadir)

    # image_dir = os.path.join(homedir, 'awork/ImageWork/imageExp1/people')
    # image_path = os.path.join(image_dir, '09.jpg')
    #
    # img = cv2.imread(image_path)
    # print('img shape: {}\tdtype: {}'.format(img.shape, img.dtype))

    # Read filenames of images we have hidden layer representations of.
    imagenames_pd = pd.read_csv(train_filename, header=0, names=['label', 'filename'])
    labels_list = imagenames_pd['label'].tolist()
    y_all = np.array(labels_list, dtype=np.int32)
    image_paths = imagenames_pd['filename'].to_list()

    # x_all, y_all, image_paths = read_labeled_images(train_filename, has_header=True)
    # x_all, y_all, image_paths = read_labeled_images(train_filename, datadir=data_dir)
    # images = x_all.copy()
    # x_all = x_all.astype('float32')
    # x_all /= 255

    # Read next-to-last hidden layer activations for known images.
    nexttolast_output = np.load(nexttolast_reps_filename)

    # Read train and test images and labels:
    x_test, y_test, test_image_paths = read_labeled_images(test_filename, datadir=data_dir)

    x_shape = x_test.shape
    n_train = y_all[0]
    nrows = x_shape[1]
    ncols = x_shape[2]
    n_channels = x_shape[3]

    n_test = x_test.shape[0]

    # Reshape the image data into 4D tensor: (sample_number, x_img_size, y_img_size, num_channels)
    # x_train = x_train.reshape(n_train, nrows, ncols, n_channels)
    # x_test = x_test.reshape(n_test, nrows, ncols, n_channels)
    input_shape = (nrows, ncols, n_channels)

    # Convert to the correct type for input to CNN:
    x_test = x_test.astype('float32')
    x_test /= 255
    print('\nAfter reshaping:')
    print('# training samples:', n_train)
    print('\nx_test shape:', x_test.shape)
    print('# test samples:', x_test.shape[0])
    print('y_test shape:', y_test.shape)
    print('y_test head:', y_test[0: 10])

    # Convert targets to one-hot vectors:
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print('y_test one-hot:\n', y_test[0:10])

    # sys.exit(0)

    # Load neural network model:
    model = keras.models.load_model(model_filename)

    # Show model architecture
    model.summary()

    # Score the model:
    score = model.evaluate(x_test, y_test, verbose=1)
    print('score:', score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Get penultimate fully-connected layer
    nexttolast_layer = model.get_layer('dense_2')
    get_nexttolast_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                             [nexttolast_layer.output])

    # Compute next-to-last hidden layer activation for the image dataset.
    # nexttolast_output = get_nexttolast_layer_output([x_all, 0])[0]
    # print('nexttolast_output shape ', nexttolast_output.shape)

    # Test: Which images are closest wrt next to last hidden layer representation?
    test_nexttolast_output = get_nexttolast_layer_output([x_test, 0])[0]
    print('test_nexttolast_output shape ', test_nexttolast_output.shape)
    np.save('_test_nexttolast_a.npy', test_nexttolast_output)

    test1 = test_nexttolast_output[0:3]
    # test1 = nexttolast_output[0:3]

    print('nexttolast_output shape: ', nexttolast_output.shape)
    print('test1 shape ', test1.shape)

    distances = distance.cdist(test1, nexttolast_output, 'euclidean')
    print('distances shape: ', distances.shape)
    distances_sorted = np.sort(distances)
    print('distances_sorted: ')
    print(distances_sorted[:, 0:5])

    distances_argsort = distances.argsort()
    lowest_distance_indices = distances_argsort[:, 0:5]

    print('image_paths shape: ', len(image_paths))

    print('Closest images to first five:')
    for i in range(test1.shape[0]):
        # fig, axs = plt.subplots(6)
        print(f'\nQuery image: {i}\t{y_all[i]}\t{image_paths[i]}')
        # print('test hidden rep')
        # print(test1[i, 0:16])
        indx_row = lowest_distance_indices[i]
        print(indx_row[0:5])
        subplot_id = 611
        plt.subplot(subplot_id)
        img = cv2.imread(image_paths[i])
        plt.imshow(img)
        plt.title('Query')
        for j in range(4):
            indx = indx_row[j]
            print(f'{j} {indx}\t{distances[i, indx]}\t{image_paths[indx]}')
            # print('search hidden rep:')
            # print(nexttolast_output[indx, 0:16])
            img = cv2.imread(image_paths[indx])
            subplot_id += 1
            plt.subplot(subplot_id)
            plt.imshow(img)
            plt.title(f'search result {j}')
        plt.savefig(f'_searchresults_{i}.png')
        plt.close()

    print('Done')
