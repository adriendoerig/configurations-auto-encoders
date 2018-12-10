# -*- coding: utf-8 -*-
"""
My script to create tfrecords files based on batchmaker class
This code is inspired by this youtube-vid and code:
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
"""

import tensorflow as tf
import itertools
import numpy as np
from batchMaker import StimMaker
from parameters import im_size, shape_size, bar_width, other_shape_ID, noise_level, tfrecords_path


##################################
#       Helper functions:        #
##################################


def wrap_bytes(value):
    output = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return output


##################################
#      tfrecords function:       #
##################################
def make_tfrecords():
    '''Function to create tfrecord files based on stim_maker class'''
    # Inputs:
    # noise: noise level
    # out_path: where to save

    print("\nConverting: " + tfrecords_path)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:

        # Create images one by one using stimMaker and save them
        stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
        matrices = np.reshape(flat_matrices, [-1, 3, 5])

        for i in range(2 ** 15):
            image, _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :] * (other_shape_ID - 1) + 1, noiseLevel=noise_level, doVernier=False)
            print("\rMaking dataset: {}/{} ({:.1f}%)".format(i, 2 ** 15, i * 100 / 2 ** 15), end="")

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


make_tfrecords()
