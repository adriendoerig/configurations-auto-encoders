from capsule_functions import *
from parameters import *

def model_fn(features, bottleneck_units, mode, LOGDIR, params):
    # images:   Our data
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   Optional parameters; here not needed because of parameter-file

    # placeholder for input images
    X = features['images']
    x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)

    if model_type is 'dense':
        with tf.name_scope('dense_auto_encoder'):
            with tf.name_scope('neurons'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                tf.summary.histogram('X_flat', X_flat)
                hidden = tf.layers.dense(X_flat, bottleneck_units, name='hidden_layer')
                tf.summary.histogram('hidden_layer', hidden)
                X_reconstructed = tf.layers.dense(hidden, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif model_type is 'dense_large':
        with tf.name_scope('dense_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                    tf.summary.histogram('X_flat', X_flat)
                    dense1 = tf.layers.dense(X_flat, n_neurons1, name='dense1')
                    tf.summary.histogram('dense1', dense1)
                    dense2 = tf.layers.dense(dense1, n_neurons2, name='dense2')
                    tf.summary.histogram('dense2', dense2)
                    encoded = tf.layers.dense(dense2, bottleneck_units, name='encoded')
                    tf.summary.histogram('encoded', encoded)
                with tf.name_scope('decoder'):
                    dense3 = tf.layers.dense(encoded, n_neurons2, name='dense3')
                    tf.summary.histogram('dense3', dense3)
                    dense4 = tf.layers.dense(dense3, n_neurons1, name='dense4')
                    tf.summary.histogram('dense4', dense4)
                    X_reconstructed = tf.layers.dense(dense4, im_size[0] * im_size[1], name='reconstruction')
                    tf.summary.histogram('X_reconstructed', X_reconstructed)
                    X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                    tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif model_type is 'conv':
        with tf.name_scope('conv_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
                tf.summary.histogram('conv2', conv2)
                conv2_flat = tf.reshape(conv2, [-1, int(np.prod(conv2.get_shape()[1:]))], name='conv2_flat')
                dense = tf.layers.dense(conv2_flat, bottleneck_units, name='dense_layer')
                tf.summary.histogram('dense', dense)
                X_reconstructed = tf.layers.dense(dense, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif model_type is 'large_conv':
        with tf.name_scope('caps_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    conv1 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=(5, 5), padding='same',
                                             activation=tf.nn.relu, name='conv1')  # Now 50x83x16
                    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                       name='pool1')  # Now 25x42x16
                    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same',
                                             activation=tf.nn.relu, name='conv2')  # Now 25x42x8
                    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                       name='pool2')  # Now 13x21/4x8
                    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=bottleneck_units, kernel_size=(3, 3),
                                             padding='same', activation=tf.nn.relu,
                                             name='conv3')  # Now 13x21xbottleneck_units
                    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                      name='pool3_encoded')  # Now 7x11xbottleneck_units
                with tf.name_scope('decoder'):
                    upsample1 = tf.image.resize_images(encoded, size=(13, 21),
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                       name='upsample1')  # Now 13x21xbottleneck_units
                    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same',
                                             activation=tf.nn.relu, name='conv4')  # Now 13x21x8
                    upsample2 = tf.image.resize_images(conv4, size=(25, 42),
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                       name='upsample2')  # Now 25x42x8
                    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same',
                                             activation=tf.nn.relu, name='conv5')  # Now 25x42x8
                    upsample3 = tf.image.resize_images(conv5, size=(50, 83),
                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                       name='upsample3')  # Now 50x83x8
                    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same',
                                             activation=tf.nn.relu, name='conv6')  # Now 50x83x16
                    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same',
                                              activation=None, name='logits')  # Now 50x83x1
                    X_reconstructed = tf.nn.sigmoid(logits,
                                                    name='X_reconstructed')  # Pass logits through sigmoid to get reconstructed image
                    X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                    tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_mean(all_losses, name='loss')
                tf.summary.scalar('loss', loss)

    elif model_type is ('caps' or 'caps_large'):
        with tf.name_scope('caps_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv1_width = int((im_size[0] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                conv1_height = int((im_size[1] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                caps1_n_caps = int((caps1_n_maps * int(
                    (conv1_width - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                    int((conv1_height - conv_caps_params["kernel_size"]) / conv_caps_params[
                                        "strides"] + 1)))
                caps1 = primary_caps_layer(conv1, caps1_n_caps, caps1_n_dims, **conv_caps_params)
                caps2 = secondary_caps_layer(caps1, caps1_n_caps, caps1_n_dims, bottleneck_units, caps2_n_dims,
                                             rba_rounds)
                caps2_flat = tf.reshape(caps2, [-1, bottleneck_units * caps2_n_dims])
                if model_type is 'caps_large':
                    dense1 = tf.layers.dense(caps2_flat, n_neurons1, name='decoder_hidden1')
                    dense2 = tf.layers.dense(dense1, n_neurons2, name='decoder_hidden2')
                    X_reconstructed = tf.layers.dense(dense2, im_size[0] * im_size[1], name='reconstruction')
                else:
                    X_reconstructed = tf.layers.dense(caps2_flat, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif model_type is 'VAE':
        tfd = tf.contrib.distributions

        def make_encoder(data, code_size):
            x = tf.layers.flatten(data, name='flatten')
            x = tf.layers.dense(x, 200, tf.nn.relu, name='encoder_dense1')
            x = tf.layers.dense(x, 200, tf.nn.relu, name='encoder_dense2')
            loc = tf.layers.dense(x, code_size, name='encoded_mu')
            scale = tf.layers.dense(x, code_size, tf.nn.softplus, name='encoded_sigma')
            return tfd.MultivariateNormalDiag(loc, scale)

        def make_prior(code_size):
            loc = tf.zeros(code_size, name='prior_mu')
            scale = tf.ones(code_size, name='prior_sigma')
            return tfd.MultivariateNormalDiag(loc, scale, name='prior_distribution')

        def make_decoder(code, data_shape):
            x = code
            x = tf.layers.dense(x, n_neurons1, tf.nn.relu, name='decoder_dense1')
            x = tf.layers.dense(x, n_neurons2, tf.nn.relu, name='decoder_dense2')
            logit = tf.layers.dense(x, np.prod(data_shape), name='decoder_logit')
            logit = tf.reshape(logit, [-1] + data_shape, name='reshapes_decoder_logit')
            return tfd.Independent(tfd.Bernoulli(logit), 2, name='decoded distribution')

        make_encoder = tf.make_template('encoder', make_encoder)
        make_decoder = tf.make_template('decoder', make_decoder)

        # Define the model.
        with tf.name_scope('prior'):
            prior = make_prior(code_size=bottleneck_units)
        with tf.name_scope('encoder'):
            posterior = make_encoder(X, code_size=bottleneck_units)
            code = posterior.sample()
        # Define the loss.
        with tf.name_scope('loss'):
            likelihood = make_decoder(code, [im_size[0], im_size[1], 1]).log_prob(X)
            divergence = tfd.kl_divergence(posterior, prior)
            all_losses = likelihood - beta * divergence
            print('careful here: correct shapes???')
            print('all_losses shape: ' + str(all_losses))
            loss = -tf.reduce_mean(all_losses)
            print('loss shape: ' + str(loss))


    # optimizer and and training operation
    with tf.name_scope('optimizer_and_training'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

    # write summaries during evaluation
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=10000, output_dir=LOGDIR + '/eval', summary_op=tf.summary.merge_all())

    # Wrap all of this in an EstimatorSpec.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # the following line is
        predictions = {'all_losses': all_losses, 'reconstructions': X_reconstructed_image}
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
        return spec

    else:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=training_op,
            eval_metric_ops={},
            evaluation_hooks=[eval_summary_hook])
        return spec
