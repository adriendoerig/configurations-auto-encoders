# -*- coding: utf-8 -*-
"""
My script to create tfrecords files based on batchmaker class
This code is inspired by this youtube-vid and code:
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
"""

import tensorflow as tf
import itertools, os
import numpy as np
from batchMaker import StimMaker
from parameters import use_these_params
if use_these_params:
    from parameters import im_size, shape_size, bar_width, other_shape_ID, noise_level, tfrecords_path_train, tfrecords_path_test
else:
    from ae_master_all_models_params import im_size, shape_size, bar_width, other_shape_ID, noise_level, tfrecords_path_train


##################################
#       Helper functions:        #
##################################


def wrap_bytes(value):
    output = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return output


##################################
#      tfrecords function:       #
##################################
def make_tfrecords(set_type):
    '''Function to create tfrecord files based on stim_maker class'''
    # set_type = 'train' or 'test'

    tf_records_path_test = './dataset_test.tfrecords'  # not used for now
    if set_type is 'train':
        save_path = tfrecords_path_train
        print("\nConverting: " + tfrecords_path_train)
    elif set_type is 'test':
        save_path = tf_records_path_test
        print("\nConverting: " + tfrecords_path_test)
    else:
        raise ValueError('set_type must be "train" or "test"')

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(save_path) as writer:

        # Create images one by one using stimMaker and save them
        stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
        matrices = np.reshape(flat_matrices, [-1, 3, 5])
        n_matrices = 2**15
        if set_type is 'test':
            flat_matrices[:, 7] = 0
            unique_flat_matrices = np.unique(flat_matrices, axis=0)
            matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
            matrices[:, 1, 2] = 0
            n_matrices = 2**14

        for i in range(n_matrices):
            image, _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :] * (other_shape_ID - 1) + 1, noiseLevel=noise_level, doVernier=False)
            print("\rMaking dataset: {}/{} ({:.1f}%)".format(i, n_matrices, i * 100 / n_matrices), end="")

            # Convert the image to raw bytes.
            image_bytes = image.tostring()

            # Create a dict with the data to save in the TFRecords file
            data = {'images': wrap_bytes(image_bytes)}

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
    return

if not os.path.exists(tfrecords_path_train):
    make_tfrecords('train')
# make_tfrecords('test')
