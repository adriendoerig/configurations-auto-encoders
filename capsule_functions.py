import tensorflow as tf
import numpy as np
from parameters import *

# define the squash function (to apply to capsule vectors)
# a safe-norm is implemented to avoid 0 norms because they
# would fuck up the gradients etc.
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm_squash = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm_squash
    return squash_factor * unit_vector

# takes the first regular convolutional layers' output as input and creates the first capsules
# returns the flattened output of the primary capsule layer (only works to feed to a FC caps layer)
def primary_caps_layer(conv_output, caps1_n_caps, caps1_n_dims, **conv_params):

    with tf.name_scope('primary_capsules'):
        conv_for_caps = tf.layers.conv2d(conv_output, name="conv_for_caps", **conv_params)
        caps1_raw = tf.reshape(conv_for_caps, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
        caps1_output = squash(caps1_raw, name="caps1_output")

    return caps1_output


# takes a (flattened) primary capsule layer caps1 output as input and creates a new fully connected capsule layer caps2
def secondary_caps_layer(caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, rba_rounds=3):

    with tf.name_scope('secondary_caps_layer'):
        # initialise weights
        init_sigma = 0.01  # stdev of weights
        W_init = lambda: tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, dtype=tf.float32, name="W")
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")  # expand last dimension
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")  # expand third dimension
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")  # tile
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")

        with tf.name_scope('routing_by_agreement'):

            def do_routing_cond(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):
                return tf.less(rba_iter, max_iter+1, name='do_routing_cond')

            def routing_by_agreement(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):

                routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
                weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
                weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
                caps2_rba_output = squash(weighted_sum, axis=-2, name="caps2_rba_output")
                caps2_rba_output_tiled = tf.tile(caps2_rba_output, [1, caps1_n_caps, 1, 1, 1], name="caps2_rba_output_tiled")
                agreement = tf.matmul(caps2_predicted, caps2_rba_output_tiled, transpose_a=True, name="agreement")
                raw_weights_new = tf.add(raw_weights, agreement, name="raw_weights_round_new")

                return caps2_predicted, caps2_rba_output, raw_weights_new, tf.add(rba_iter, 1)

            # initialize routing weights
            raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
            caps2_predicted, caps2_output, raw_weights, rba_iter = tf.while_loop(do_routing_cond, routing_by_agreement, [caps2_predicted, caps2_output, raw_weights, rba_iter])

        return caps2_output
