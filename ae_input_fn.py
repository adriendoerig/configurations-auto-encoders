# -*- coding: utf-8 -*-
"""
My script for the input fn using tfrecords files (good for estimators)
"""

import tensorflow as tf
import numpy as np
from parameters import im_size, n_epochs, batch_size, tfrecords_path


########################################
#     Parse tfrecords training set:    #
########################################
def parse_tfrecords(serialized_data):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    features = {'images': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    images = parsed_data['shape_images']
    images = tf.decode_raw(images, tf.float32)
    images = tf.cast(images, tf.float32)

    # Reshaping:
    images = tf.reshape(images, [im_size[0], im_size[1], 1])

    # Get rid of extra-dimension:
    print(images)
    images = tf.squeeze(images, axis=0)
    print(images)

    return images


###########################
#     Input function:     #
###########################
def input_fn(filenames, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)

    dataset = dataset.map(parse_tfrecords, num_parallel_calls=64)

    # Read a buffer of the given size and randomly shuffle it:
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Allow for infinite reading of data
    num_repeat = n_epochs

    # Repeat the dataset the given number of times and get a batch of data
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images = iterator.get_next()

    feed_dict = {'images': images}

    return feed_dict


##############################
#   Final input functions:   #
##############################
def input_fn():
    return input_fn(filenames=tfrecords_path)
