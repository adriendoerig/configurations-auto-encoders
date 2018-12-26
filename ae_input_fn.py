# -*- coding: utf-8 -*-
"""
My script for the input fn using tfrecords files (good for estimators)
"""

import tensorflow as tf
from parameters import use_these_params
if use_these_params:
    from parameters import im_size, n_epochs, batch_size, tfrecords_path_train
else:
    from ae_master_all_models_params import im_size, n_epochs, batch_size, tfrecords_path_train


########################################
#     Parse tfrecords training set:    #
########################################
def parse_tfrecords(serialized_data):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    features = {'images': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    images = parsed_data['images']
    images = tf.decode_raw(images, tf.float32)
    images = tf.cast(images, tf.float32)

    # Reshaping:
    images = tf.reshape(images, [im_size[0], im_size[1], 1])

    return images


###########################
#     Input function:     #
###########################
def input_fn_train(filenames, buffer_size=2**15):
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


# the following function makes tf.dataset_tests from numpy batches. if latent_input is not None, also provides a latent input to decode from
def input_fn_pred(batch, latent_input=None, return_batch_size=False):

    batch_size = batch.shape[0]
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    if latent_input is None:
        if return_batch_size:
            batch_sizes = tf.convert_to_tensor(batch_size, dtype=tf.int64)
            batch_sizes = tf.tile([batch_sizes], [batch_size])  # inelegant, but the input_fn must return a vector of batch sizes even if we only use one in the end.
            dataset_test = tf.data.Dataset.from_tensor_slices((batch, batch_sizes))
        else:
            dataset_test = tf.data.Dataset.from_tensor_slices(batch)
    else:
        if return_batch_size:
            batch_sizes = tf.convert_to_tensor(batch_size, dtype=tf.int64)
            batch_sizes = tf.tile([batch_sizes], [batch_size])
            user_latent_input = tf.convert_to_tensor(latent_input, dtype=tf.float32)
            dataset_test = tf.data.Dataset.from_tensor_slices((batch, user_latent_input, batch_sizes))
        else:
            user_latent_input = tf.convert_to_tensor(latent_input, dtype=tf.float32)
            dataset_test = tf.data.Dataset.from_tensor_slices((batch, user_latent_input))

    dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset_test = dataset_test.prefetch(2)
    # Create an iterator for the dataset_test and the above modifications.
    iterator = dataset_test.make_one_shot_iterator()

    # Get the next batch of images and labels.
    if latent_input is None:
        if return_batch_size:
            [images, batch_sizes] = iterator.get_next()
            feed_dict = {'images': images,
                         'batch_sizes': batch_sizes}
        else:
            images = iterator.get_next()
            feed_dict = {'images': images}
    else:
        if return_batch_size:
            [images, user_latent_input, batch_sizes] = iterator.get_next()
            feed_dict = {'images': images,
                         'user_latent_input': user_latent_input,
                         'batch_sizes': batch_sizes}
        else:
            [images, user_latent_input] = iterator.get_next()
            feed_dict = {'images': images,
                         'user_latent_input': user_latent_input}
    return feed_dict


##############################
#   Final input functions:   #
##############################
def input_fn():
    return input_fn_train(filenames=tfrecords_path_train)

