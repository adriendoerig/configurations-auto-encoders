from capsule_functions import *
from parameters import use_these_params
if use_these_params:
    from parameters import im_size
else:
    from ae_master_all_models_params import im_size


def model_fn(features, labels, mode, params):

    # placeholder for input images
    X = features['images']
    x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)
    if 'user_latent_input' in features:
        user_latent_input = features['user_latent_input']
    else:
        user_latent_input = None
    if 'batch_sizes' in features:  # NOTE: i think this is stupid. the goal was to flexibly define batch_size online, but i don't think it works. it just uses always the same batch size as the first time the network is created... so always the batch_size defined in parameters (although I am not certain since, for example, the capsules embeddings indeed seems to have all the configurations and not just 64)
        batch_sizes = features['batch_sizes']
        batch_size = batch_sizes[0]  # inelegant, but the input_fn must return a vector of batch sizes even if we only use one in the end.
    else:
        if use_these_params:
            from parameters import batch_size
        else:
            from ae_master_all_models_params import batch_size

    if params['model_type'] is 'dense':
        with tf.name_scope('dense_auto_encoder'):
            with tf.name_scope('neurons'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                tf.summary.histogram('X_flat', X_flat)
                encoded = tf.layers.dense(X_flat, params['bottleneck_units'], name='hidden_layer')
                tf.summary.histogram('hidden_layer', encoded)
                if user_latent_input is not None:
                    encoded = user_latent_input
                X_reconstructed = tf.layers.dense(encoded, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif params['model_type'] is 'large_dense':

        n_neurons1 = 50
        n_neurons2 = 50

        with tf.name_scope('large_dense_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                    tf.summary.histogram('X_flat', X_flat)
                    dense1 = tf.layers.dense(X_flat, n_neurons1, name='dense1')
                    tf.summary.histogram('dense1', dense1)
                    dense2 = tf.layers.dense(dense1, n_neurons2, name='dense2')
                    tf.summary.histogram('dense2', dense2)
                    encoded = tf.layers.dense(dense2, params['bottleneck_units'], name='encoded')
                    tf.summary.histogram('encoded', encoded)
                    if user_latent_input is not None:
                        encoded = user_latent_input
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

    elif params['model_type'] is 'conv':
        conv_activation_function = tf.nn.elu
        conv1_params = {"filters": 64,
                        "kernel_size": 11,
                        "strides": 1,
                        "padding": "valid",
                        "activation": conv_activation_function,
                        }
        conv2_params = {"filters": 64,
                        "kernel_size": 10,
                        "strides": 2,
                        "padding": "valid",
                        "activation": conv_activation_function,
                        }
        with tf.name_scope('conv_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
                tf.summary.histogram('conv2', conv2)
                conv3 = tf.reshape(conv2, [-1, int(np.prod(conv2.get_shape()[1:]))], name='conv2_flat')
                encoded = tf.layers.dense(conv3, params['bottleneck_units'], name='dense_layer')
                tf.summary.histogram('encoded', encoded)
                if user_latent_input is not None:
                    encoded = user_latent_input
                X_reconstructed = tf.layers.dense(encoded, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif params['model_type'] is 'large_conv':
        # cf https://github.com/mchablani/deep-learning/blob/master/autoencoder/Convolutional_Autoencoder.ipynb
        with tf.name_scope('large_conv_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    # NOTE: the layer sizes marked below are for 32x52 input images
                    conv1 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv1')  # Now 32x52x16
                    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')  # Now 16x26x16
                    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2')  # Now 16x26x8
                    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')  # Now 8x13/4x8
                    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=params['bottleneck_units'], kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3')  # Now 8x13xbottleneck_units
                    maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')  # Now 4x7xbottleneck_units
                with tf.name_scope('decoder'):
                    # NOTE: the layer sizes marked below are for 32x52 input images
                    maxpool3_flat = tf.layers.flatten(maxpool3, name='maxpool3_flat')
                    encoded = tf.layers.dense(maxpool3_flat, params['bottleneck_units'], name='encoded')
                    if user_latent_input is not None:
                        encoded = user_latent_input
                    upsampled_code = tf.layers.dense(encoded, 4 * 7 * params['bottleneck_units'], name='upsampled_code')  # (4,7) is the shape of the last maxpool(conv) layer in the encoder
                    reshaped_upsampled_code = tf.reshape(upsampled_code, [-1, 4, 7, params['bottleneck_units']], name='reshaped_upsampled_code')
                    upsample1 = tf.image.resize_images(reshaped_upsampled_code, size=(8, 13), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 8x13xbottleneck_units
                    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4')  # Now 8x13x8
                    upsample2 = tf.image.resize_images(conv4, size=(16, 26), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 16x26x8
                    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5')  # Now 16x26x8
                    upsample3 = tf.image.resize_images(conv5, size=(32, 52), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 32x52x8
                    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv6')  # Now 32x52x16
                    X_reconstructed = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None, name='logits')  # Now 32x52x1
                    X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                    tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                X_reconstructed_flat = tf.reshape(X_reconstructed, [-1, im_size[0] * im_size[1]], name='X_reconstructed_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed_flat, X_flat, name='square_diffs'), axis=1,
                                           name='losses_per_image')
                loss = tf.reduce_mean(all_losses, name='loss')
                tf.summary.scalar('loss', loss)

    elif 'caps' in params['model_type']:
        # conv layers
        activation_function = tf.nn.elu
        conv1_params = {"filters": 64,
                        "kernel_size": 11,
                        "strides": 1,
                        "padding": "valid",
                        "activation": activation_function,
                        }
        # primary capsules
        caps1_n_maps = 8  # number of capsules at level 1 of capsules
        caps1_n_dims = 8  # number of dimension per capsule (note: 8*8=64 to have the same number of neurons as the convnet)
        conv_caps_params = {"filters": caps1_n_maps * caps1_n_dims,
                            "kernel_size": 10,
                            "strides": 2,
                            "padding": "valid",
                            "activation": activation_function,
                            }
        # output capsules
        if '16_dims' in params['model_type']:
            caps2_n_dims = 16
        else:
            caps2_n_dims = 4  # of n dimensions

        rba_rounds = 3
        if model_type is 'large_caps':
            n_neurons1 = 50
            n_neurons2 = 50

        with tf.name_scope('caps_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv1_width = int((im_size[0] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                conv1_height = int((im_size[1] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                caps1_n_caps = int((caps1_n_maps * int((conv1_width - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                                   int((conv1_height - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1)))
                caps1 = primary_caps_layer(conv1, caps1_n_caps, caps1_n_dims, **conv_caps_params)
                caps2 = secondary_caps_layer(caps1, caps1_n_caps, caps1_n_dims, params['bottleneck_units'], caps2_n_dims, rba_rounds, batch_size=batch_size)
                encoded = tf.reshape(caps2, [-1, params['bottleneck_units'] * caps2_n_dims])
                if user_latent_input is not None:
                    encoded = user_latent_input
                if params['model_type'] is 'large_caps':
                    dense1 = tf.layers.dense(encoded, n_neurons1, name='decoder_hidden1')
                    dense2 = tf.layers.dense(dense1, n_neurons2, name='decoder_hidden2')
                    X_reconstructed = tf.layers.dense(dense2, im_size[0] * im_size[1], name='reconstruction')
                else:
                    X_reconstructed = tf.layers.dense(encoded, im_size[0] * im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)

    elif params['model_type'] is 'VAE' or params['model_type'] is 'VAE_beta2':
        import tensorflow_probability as tfp
        tfd = tfp.distributions

        # cf. https://danijar.com/building-variational-auto-encoders-in-tensorflow/
        # and https://colab.research.google.com/drive/1Wl78KHPzQ2Q253Rob5W1o0bd8nu9DZel#scrollTo=zaCO7S0-_KNn&forceEdit=true&offline=true&sandboxMode=true
        # good explanation https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
        n_neurons1 = 50
        n_neurons2 = 50

        if params['model_type'] is 'VAE':
            beta = 1  # for disentangled representations: >1
        elif params['model_type'] is 'VAE_beta2':
            beta = 2

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
            return tfd.Independent(tfd.Bernoulli(logit), 3, name='decoded_distribution')

        make_encoder = tf.make_template('encoder', make_encoder)
        make_decoder = tf.make_template('decoder', make_decoder)

        # Define the model.
        with tf.name_scope('prior'):
            prior = make_prior(code_size=params['bottleneck_units'])
        with tf.name_scope('encoder'):
            posterior = make_encoder(X, code_size=params['bottleneck_units'])
            encoded = posterior.sample()
            if user_latent_input is not None:
                encoded = user_latent_input
        # Define the loss.
        with tf.name_scope('loss'):
            likelihood = make_decoder(encoded, [im_size[0], im_size[1], 1]).log_prob(X)
            divergence = tfd.kl_divergence(posterior, prior)
            all_losses = -(likelihood - beta * divergence)
            loss = tf.reduce_mean(all_losses)
            tf.summary.scalar('loss', loss)
        with tf.name_scope('reconstructions'):
            X_reconstructed_image = make_decoder(encoded, [im_size[0], im_size[1], 1]).mean()
            tf.summary.image('reconstructions', X_reconstructed_image, 6)
            samples = make_decoder(prior.sample(6), [im_size[0], im_size[1], 1]).mean()
            tf.summary.image('samples', samples, 6)

    elif params['model_type'] is 'VAE_conv' or params['model_type'] is 'VAE_conv_beta2':
        import tensorflow_probability as tfp
        tfd = tfp.distributions

        if params['model_type'] is 'VAE_conv':
            beta = 1  # for disentangled representations: >1
        elif params['model_type'] is 'VAE_conv_beta2':
            beta = 2

        def make_encoder(data, code_size):
            # NOTE: the layer sizes marked below are for 32x52 input images
            conv1 = tf.layers.conv2d(inputs=data, filters=16, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv1')  # Now 32x52x16
            maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')  # Now 16x26x16
            conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2')  # Now 16x26x8
            maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')  # Now 8x13/4x8
            conv3 = tf.layers.conv2d(inputs=maxpool2, filters=params['bottleneck_units'], kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3')  # Now 8x13xbottleneck_units
            maxpool3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')  # Now 4x7xbottleneck_units
            maxpool3_flat = tf.reshape(maxpool3, [-1, 4*7*params['bottleneck_units']])
            loc = tf.layers.dense(maxpool3_flat, code_size, name='encoded_mu')
            scale = tf.layers.dense(maxpool3_flat, code_size, tf.nn.softplus, name='encoded_sigma')
            return tfd.MultivariateNormalDiag(loc, scale)

        def make_prior(code_size):
            loc = tf.zeros(code_size, name='prior_mu')
            scale = tf.ones(code_size, name='prior_sigma')
            return tfd.MultivariateNormalDiag(loc, scale, name='prior_distribution')

        def make_decoder(code):
            # NOTE: the layer sizes marked below are for 32x52 input images
            upsampled_code = tf.layers.dense(code, 4*7*params['bottleneck_units'], name='upsampled_code')  # (4,7) is the shape of the last maxpool(conv) layer in the encoder
            reshaped_upsampled_code = tf.reshape(upsampled_code, [-1, 4, 7, params['bottleneck_units']], name='reshaped_upsampled_code')
            upsample1 = tf.image.resize_images(reshaped_upsampled_code, size=(8, 13), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 8x13xbottleneck_units
            conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4')  # Now 8x13x8
            upsample2 = tf.image.resize_images(conv4, size=(16, 26), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 16x26x8
            conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5')  # Now 16x26x8
            upsample3 = tf.image.resize_images(conv5, size=(32, 52), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 32x52x8
            conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv6')  # Now 32x52x16
            logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None, name='logits')  # Now 32x52x1
            return tfd.Independent(tfd.Bernoulli(logits), 3, name='decoded_distribution')

        make_encoder = tf.make_template('encoder', make_encoder)
        make_decoder = tf.make_template('decoder', make_decoder)

        # Define the model.
        with tf.name_scope('prior'):
            prior = make_prior(code_size=params['bottleneck_units'])
        with tf.name_scope('encoder'):
            posterior = make_encoder(X, code_size=params['bottleneck_units'])
            encoded = posterior.sample()
            if user_latent_input is not None:
                encoded = user_latent_input
        # Define the loss.
        with tf.name_scope('loss'):
            likelihood = make_decoder(encoded).log_prob(X)
            divergence = tfd.kl_divergence(posterior, prior)
            all_losses = -(likelihood - beta * divergence)
            loss = tf.reduce_mean(all_losses)
            tf.summary.scalar('loss', loss)
        with tf.name_scope('reconstructions'):
            X_reconstructed_image = make_decoder(encoded).mean()
            tf.summary.image('reconstructions', X_reconstructed_image, 6)
            samples = make_decoder(prior.sample(6)).mean()
            tf.summary.image('samples', samples, 6)

    elif params['model_type'] is 'alexnet_layers_1_3' or params['model_type'] is 'alexnet_layers_1_5':
        # alexnet + weights comes from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

        # pad image to alexnet size
        def pad_up_to(t, max_in_dims, constant_values):
            s = tf.shape(t)
            paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
            return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)
        # Alexnet takes larger 227*227 images. we zoom into our dataset and pad with zeros until we get the right size
        zoom = 4
        if 'batch_sizes' in features:
            with tf.Session() as sess:
                np_batch_size = batch_size.eval()
        else:
            np_batch_size = batch_size
        X_alexnet = pad_up_to(tf.image.resize_images(X, size=[im_size[0]*zoom, im_size[1]*zoom], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True), [np_batch_size, 227, 227, 1], 0)
        # we tile to have the 3 rgb channels expected by the model
        X_alexnet = tf.tile(X_alexnet, [1, 1, 1, 3])
        tf.summary.image('resized_images', X_alexnet, 6)

        net_data = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()

        # helper function used in http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ 's version of alexnet
        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = input.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
                kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)

            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])


        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        with tf.name_scope('encoder'):
            with tf.name_scope('conv1'):
                k_h = 11
                k_w = 11
                c_o = 96
                s_h = 4
                s_w = 4
                conv1W = tf.Variable(net_data["conv1"][0])
                conv1b = tf.Variable(net_data["conv1"][1])
                conv1_in = conv(X_alexnet, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
                conv1 = tf.nn.relu(conv1_in)
                tf.summary.histogram('conv1', conv1)

            # lrn1
            # lrn(2, 2e-05, 0.75, name='norm1')
            with tf.name_scope('lrn1'):
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

            # maxpool1
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            with tf.name_scope('maxpool1'):
                k_h = 3
                k_w = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv2
            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            with tf.name_scope('conv2'):
                k_h = 5
                k_w = 5
                c_o = 256
                s_h = 1
                s_w = 1
                group = 2
                conv2W = tf.Variable(net_data["conv2"][0])
                conv2b = tf.Variable(net_data["conv2"][1])
                conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
                conv2 = tf.nn.relu(conv2_in)
                tf.summary.histogram('conv2', conv2)

            # lrn2
            # lrn(2, 2e-05, 0.75, name='norm2')
            with tf.name_scope('lrn2'):
                radius = 2
                alpha = 2e-05
                beta = 0.75
                bias = 1.0
                lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

            # maxpool2
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            with tf.name_scope('maxpool2'):
                k_h = 3
                k_w = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv3
            # conv(3, 3, 384, 1, 1, name='conv3')
            with tf.name_scope('conv3'):
                k_h = 3
                k_w = 3
                c_o = 384
                s_h = 1
                s_w = 1
                group = 1
                conv3W = tf.Variable(net_data["conv3"][0])
                conv3b = tf.Variable(net_data["conv3"][1])
                conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
                conv3 = tf.nn.relu(conv3_in)
                tf.summary.histogram('conv3', conv3)

            # conv4
            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            with tf.name_scope('conv4'):
                k_h = 3
                k_w = 3
                c_o = 384
                s_h = 1
                s_w = 1
                group = 2
                conv4W = tf.Variable(net_data["conv4"][0])
                conv4b = tf.Variable(net_data["conv4"][1])
                conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
                conv4 = tf.nn.relu(conv4_in)
                tf.summary.histogram('conv4', conv4)

            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            with tf.name_scope('conv5'):
                k_h = 3
                k_w = 3
                c_o = 256
                s_h = 1
                s_w = 1
                group = 2
                conv5W = tf.Variable(net_data["conv5"][0])
                conv5b = tf.Variable(net_data["conv5"][1])
                conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
                conv5 = tf.nn.relu(conv5_in)
                tf.summary.histogram('conv5', conv5)

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            with tf.name_scope('maxpool5'):
                k_h = 3
                k_w = 3
                s_h = 2
                s_w = 2
                padding = 'VALID'
                maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope('alexnet_decoder'):
            if params['model_type'] is 'alexnet_layers_1_3':
                conv3_flat = tf.layers.flatten(conv3, name='conv3_flat')
                encoded = tf.layers.dense(conv3_flat, params['bottleneck_units'], name='encoded')
                if user_latent_input is not None:
                    encoded = user_latent_input
                conv3_flatinv = tf.layers.dense(encoded, 13*13*384, name='conv3_flatinv')  # (13,13,384) is the shape of the conv3 layer in the encoder
                reshaped_conv3_flatinv = tf.reshape(conv3_flatinv, [-1, 13, 13, 384], name='reshaped_upsampled_code')
                conv3inv = tf.layers.conv2d(inputs=reshaped_conv3_flatinv, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3inv')  # Now 28x28x256 (like the conv2 layer)
                upsample1 = tf.image.resize_images(conv3inv, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 28x28x384
                conv2inv = tf.layers.conv2d(inputs=upsample1, filters=96, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv2inv')  # Now 57x57x96 (like the conv1 layer
                upsample2 = tf.image.resize_images(conv2inv, size=(57, 57), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 57x57x256
                conv1inv = tf.layers.conv2d(inputs=upsample2, filters=3, kernel_size=(11, 11), padding='same', activation=tf.nn.relu, name='conv1inv')  # Now 57x57x3
                upsample3 = tf.image.resize_images(conv1inv, size=(227, 227), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 227x227x3 counters the 4x4 strides in alexnet's conv1
                X_reconstructed = tf.layers.conv2d(inputs=upsample3, filters=3, kernel_size=(4, 4), padding='same', activation=tf.nn.relu, name='conv1stridesinv')  # Now 227x227x3
                X_reconstructed_image = X_reconstructed
                tf.summary.image('reconstructions', X_reconstructed_image, 6)
            elif params['model_type'] is 'alexnet_layers_1_5':
                maxpool5_flat = tf.layers.flatten(maxpool5, name='conv3_flat')
                encoded = tf.layers.dense(maxpool5_flat, params['bottleneck_units'], name='encoded')
                if user_latent_input is not None:
                    encoded = user_latent_input
                conv5_flatinv = tf.layers.dense(encoded, 13 * 13 * 256, name='conv5_flatinv')  # (13,13,256) is the shape of the conv5 layer in the encoder
                reshaped_conv5_flatinv = tf.reshape(conv5_flatinv, [-1, 13, 13, 256], name='reshaped_conv5_flatinv')
                conv5inv = tf.layers.conv2d(inputs=reshaped_conv5_flatinv, filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5inv')  # Now 13x13x384 (like the conv4 layer)
                conv4inv = tf.layers.conv2d(inputs=conv5inv, filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4inv')  # Now 13x13x384 (like the conv3 layer)
                conv3inv = tf.layers.conv2d(inputs=conv4inv, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3inv')  # Now 13x13x256 (like the maxpool2 layer)
                upsample1 = tf.image.resize_images(conv3inv, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 28x28x384
                conv2inv = tf.layers.conv2d(inputs=upsample1, filters=96, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv2inv')  # Now 57x57x96 (like the conv1 layer)
                upsample2 = tf.image.resize_images(conv2inv, size=(57, 57), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 57x57x256
                conv1inv = tf.layers.conv2d(inputs=upsample2, filters=3, kernel_size=(11, 11), padding='same', activation=tf.nn.relu, name='conv1inv')  # Now 57x57x3
                upsample3 = tf.image.resize_images(conv1inv, size=(227, 227), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # Now 227x227x3 counters the 4x4 strides in alexnet's conv1
                X_reconstructed = tf.layers.conv2d(inputs=upsample3, filters=3, kernel_size=(4, 4), padding='same', activation=tf.nn.relu, name='conv1stridesinv')  # Now 227x227x3
                X_reconstructed_image = X_reconstructed
                tf.summary.image('reconstructions', X_reconstructed_image, 6)

        with tf.name_scope('reconstruction_loss'):
            X_flat = tf.reshape(X_alexnet, [-1, 227 * 227 * 3], name='X_flat')
            X_reconstructed_flat = tf.reshape(X_reconstructed, [-1, 227 * 227 * 3], name='X_reconstructed_flat')
            all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed_flat, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
            loss = tf.reduce_mean(all_losses, name='loss')
            tf.summary.scalar('loss', loss)

    else:
        raise ValueError('MODEL TYPE NOT RECOGNIZED.')

    # optimizer and and training operation
    with tf.name_scope('optimizer_and_training'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        if 'alexnet' in params['model_type']:
            print('Using alexnet encoder: will not train encoder weights.')
            training_op = optimizer.minimize(loss=loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='alexnet_decoder'), global_step=tf.train.get_global_step(), name="training_op")
        else:
            training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

    # write summaries during evaluation
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=10000, output_dir=params['LOGDIR'] + '/eval', summary_op=tf.summary.merge_all())

    # Wrap all of this in an EstimatorSpec.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # the following line is
        predictions = {'all_losses': all_losses, 'reconstructions': X_reconstructed_image, 'encoded':encoded}
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
