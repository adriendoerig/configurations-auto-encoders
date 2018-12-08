from batchMaker import StimMaker
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import os
from capsule_functions import primary_caps_layer, secondary_caps_layer
import itertools

########################################################################################################################
# Create dataset
########################################################################################################################

stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
if not os.path.exists('./dataset.npy'):
    flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
    matrices = np.reshape(flat_matrices, [-1, 3, 5])
    matrices[:, 1, 2] = 1
    dataset = np.zeros(shape=(matrices.shape[0], im_size[0], im_size[1], 1))  # need fourth dimension for tensorflow

    for i in range(2**14):
        dataset[i, :, :, :], _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :]*(other_shape_ID-1) + 1, doVernier=False)
        print("\rMaking dataset: {}/{} ({:.1f}%)".format(i, 2**14, i * 100 / 2**14),end="")

    np.save('dataset.npy', dataset)

else:
    print(' dataset.npy found -> loading')
    dataset = np.load('./dataset.npy')


# in case you wanna check your input
show_samples = 0
if show_samples:

    n_samples = 5

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        sample_image = dataset[np.random.randint(2**14), :, :].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
    plt.show()

# we will do a loop over many diffent number of hidden units
if model_type is 'caps':  # we don't use ALL n_hidden_units. Here, choose which ones to use.
    chosen_n_units = range(1, n_hidden_units_max + 1)
elif model_type is 'large_conv':
    chosen_n_units = range(1, bottleneck_features_max + 1)
else:
    chosen_n_units = range(8, n_hidden_units_max + 1, 4)

final_losses_order_all = np.zeros(shape=(len(chosen_n_units), 2**14))
run_ID = 0

for n_hidden_units in chosen_n_units:

    print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
    ### LOGDIR  ###
    LOGDIR = './' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
    checkpoint_path = LOGDIR + '/checkpoint.ckpt'

    ########################################################################################################################
    # Create Network
    ########################################################################################################################

    tf.reset_default_graph()

    # placeholder for input images
    X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
    x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)

    if model_type is 'dense':
        with tf.name_scope('dense_auto_encoder'):
            with tf.name_scope('neurons'):
                X_flat = tf.reshape(X, [-1, im_size[0]*im_size[1]], name='X_flat')
                tf.summary.histogram('X_flat', X_flat)
                hidden = tf.layers.dense(X_flat, n_hidden_units, name='hidden_layer')
                tf.summary.histogram('hidden_layer', hidden)
                X_reconstructed = tf.layers.dense(hidden, im_size[0]*im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)
            with tf.name_scope('optimizer_and_training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
    elif model_type is 'dense_large':
        with tf.name_scope('dense_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    X_flat = tf.reshape(X, [-1, im_size[0]*im_size[1]], name='X_flat')
                    tf.summary.histogram('X_flat', X_flat)
                    dense1 = tf.layers.dense(X_flat, n_neurons1, name='dense1')
                    tf.summary.histogram('dense1', dense1)
                    dense2 = tf.layers.dense(dense1, n_neurons2, name='dense2')
                    tf.summary.histogram('dense2', dense2)
                    encoded = tf.layers.dense(dense2, n_hidden_units, name='encoded')
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
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)
            with tf.name_scope('optimizer_and_training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
    elif model_type is 'conv':
        with tf.name_scope('conv_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
                tf.summary.histogram('conv2', conv2)
                conv2_flat = tf.reshape(conv2, [-1, int(np.prod(conv2.get_shape()[1:]))], name='conv2_flat')
                dense = tf.layers.dense(conv2_flat, n_hidden_units, name='dense_layer')
                tf.summary.histogram('dense', dense)
                X_reconstructed = tf.layers.dense(dense, im_size[0]*im_size[1], name='reconstruction')
                tf.summary.histogram('X_reconstructed', X_reconstructed)
                X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)
            with tf.name_scope('optimizer_and_training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
    elif model_type is 'large_conv':
        with tf.name_scope('caps_auto_encoder'):
            with tf.name_scope('neurons'):
                with tf.name_scope('encoder'):
                    conv1 = tf.layers.conv2d(inputs=X, filters=16, kernel_size=(5, 5), padding='same', activation=tf.nn.relu, name='conv1')  # Now 50x83x16
                    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')  # Now 25x42x16
                    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv2')  # Now 25x42x8
                    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')  # Now 13x21/4x8
                    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=n_hidden_units, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv3')  # Now 13x21xn_hidden_units
                    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3_encoded')  # Now 7x11xn_hidden_units
                with tf.name_scope('decoder'):
                    upsample1 = tf.image.resize_images(encoded, size=(13, 21), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='upsample1')  # Now 13x21xn_hidden_units
                    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv4')  # Now 13x21x8
                    upsample2 = tf.image.resize_images(conv4, size=(25, 42), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='upsample2')  # Now 25x42x8
                    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv5')  # Now 25x42x8
                    upsample3 = tf.image.resize_images(conv5, size=(50, 83), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='upsample3')  # Now 50x83x8
                    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, name='conv6')  # Now 50x83x16
                    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None, name='logits')  # Now 50x83x1
                    X_reconstructed = tf.nn.sigmoid(logits, name='X_reconstructed')  # Pass logits through sigmoid to get reconstructed image
                    X_reconstructed_image = tf.reshape(X_reconstructed, [-1, im_size[0], im_size[1], 1])
                    tf.summary.image('reconstruction', X_reconstructed_image, 6)
            with tf.name_scope('reconstruction_loss'):
                X_flat = tf.reshape(X, [-1, im_size[0] * im_size[1]], name='X_flat')
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_mean(all_losses, name='loss')
                tf.summary.scalar('loss', loss)
            with tf.name_scope('optimizer_and_training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
    elif model_type is ('caps' or 'caps_large'):
        with tf.name_scope('caps_auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv1_width = int((im_size[0] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                conv1_height = int((im_size[1] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
                caps1_n_caps = int((caps1_n_maps * int((conv1_width  - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                                   int((conv1_height - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1)))
                caps1 = primary_caps_layer(conv1, caps1_n_caps, caps1_n_dims, **conv_caps_params)
                caps2 = secondary_caps_layer(caps1, caps1_n_caps, caps1_n_dims, n_hidden_units, caps2_n_dims, rba_rounds)
                caps2_flat = tf.reshape(caps2, [-1, n_hidden_units*caps2_n_dims])
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
                all_losses = tf.reduce_sum(tf.squared_difference(X_reconstructed, X_flat, name='square_diffs'), axis=1, name='losses_per_image')
                loss = tf.reduce_sum(all_losses, name='total_loss')
                tf.summary.scalar('loss', loss)
            with tf.name_scope('optimizer_and_training'):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
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
        def plot_online(epoch, codes, samples, size=1):
            fig, ax = plt.subplots(
                ncols=1 + len(samples), figsize=((1 + len(samples)) * size, size))
            no_ticks = dict(left='off', bottom='off', labelleft='off', labelbottom='off')
            ax[0].set_ylabel('Epoch {}'.format(epoch))
            ax[0].scatter(codes[:, 0], codes[:, 1], s=2, alpha=0.1)
            ax[0].set_aspect('equal')
            ax[0].set_xlim(codes.min() - .1, codes.max() + .1)
            ax[0].set_ylim(codes.min() - .1, codes.max() + .1)
            ax[0].tick_params(axis='both', which='both', **no_ticks)
            for index, sample in enumerate(samples):
                ax[1 + index].imshow(sample, cmap='gray')
                ax[1 + index].tick_params(axis='both', which='both', **no_ticks)
            plt.savefig('./results/' + model_type + '_online_plot_epoch_' + str(epoch) + '.png')

        make_encoder = tf.make_template('encoder', make_encoder)
        make_decoder = tf.make_template('decoder', make_decoder)

        # Define the model.
        with tf.name_scope('prior'):
            prior = make_prior(code_size=n_hidden_units)
        with tf.name_scope('encoder'):
            posterior = make_encoder(X, code_size=n_hidden_units)
            code = posterior.sample()
        # Define the loss.
        with tf.name_scope('loss & optimizer'):
            likelihood = make_decoder(code, [im_size[0], im_size[1], 1]).log_prob(X)
            divergence = tfd.kl_divergence(posterior, prior)
            all_losses = likelihood - beta*divergence
            print('careful here: correct shapes???')
            print('all_losses shape: ' + str(all_losses))
            loss = -tf.reduce_mean(all_losses)
            print('loss shape: ' + str(loss))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
        # getting samples
        with tf.name_scope('sample_generator'):
            samples = make_decoder(prior.sample(10), [28, 28]).mean()

    ########################################################################################################################
    # Training
    ########################################################################################################################

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
    n_batches = 2**14//batch_size

    with tf.Session() as sess:

        writer = tf.summary.FileWriter(LOGDIR, sess.graph)
        # tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

        if tf.train.checkpoint_exists(checkpoint_path):
            saver.restore(sess, checkpoint_path)
            print('Capser checkpoint found. Skipping training.')

        else:
            init.run()
            print('Training')

            for epoch in range(n_epochs):

                if model_type is 'VAE':
                    # For VAE, we do some online scatter plotting each epoch (plots saved in 'VAE' folder).
                    batch_data = dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :].shape)
                    plot_loss, plot_codes, plot_samples = sess.run([loss, code, samples], {X: batch_data})
                    plot_online(epoch, plot_codes, plot_samples)

                # shuffle data
                np.random.shuffle(dataset)
                for batch in range(n_batches):
                    # get new batch
                    batch_data = dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :].shape)

                    # Run the training operation and measure the loss:
                    _, loss_train, summ = sess.run([training_op, loss, summary], feed_dict={X: batch_data})

                    if batch % 50 == 0:
                        writer.add_summary(summ, epoch*n_batches+batch)

                    print("\rEpoch: {}/{} - Batch: {}/{} ({:.1f}%) Total loss: {:.5f}".format(
                        epoch+1, n_epochs, batch+1, n_batches, (batch+1) * 100 / n_batches, loss_train), end="")

            # do final embeddings and save network
            # sess.run([assignment_images, assignment_hidden, assignment_reconstructions], feed_dict={X: embedding_data})
            # sess.run(assignment_hidden, feed_dict={X: embedding_data})  # only hidden layer embeddings (faster)
            saver.save(sess, checkpoint_path)


    ########################################################################################################################
    # Classify the reconstructed images from best to worst
    ########################################################################################################################


    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        # reload dataset so it is always in the same order
        dataset = np.load('./dataset.npy')

        n_trials = 10
        final_losses = np.zeros(shape=(n_trials, 2**14))
        final_reconstructions = np.zeros(shape=(2**14, im_size[0], im_size[1], 1))
        for trial in range(n_trials):
            for batch in range(n_batches):
                # get all post-training losses
                final_losses[trial, batch*batch_size:batch*batch_size+batch_size],  final_reconstructions[batch*batch_size:batch*batch_size+batch_size, :, :, :] \
                    = sess.run([all_losses, X_reconstructed_image],
                               feed_dict={X: dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset[batch*batch_size:batch*batch_size+batch_size, :, :, :].shape)})

    final_losses = np.mean(final_losses, axis=0)

    # get indices of the configurations from lowest to highest loss
    final_losses_order = final_losses.argsort()
    final_losses_order_all[run_ID, :] = final_losses_order  # the -1 because the loop goes from 1->n_hidden_units_max, but indexing starts at 0

    # show the first few best images
    n_samples = 5
    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(2, n_samples, index + 1)
        sample_image = dataset[final_losses_order[index], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title(('Rank: ' + str(index)))
        plt.subplot(2, n_samples, n_samples + index + 1)
        sample_image = final_reconstructions[final_losses_order[index], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[index]])))
    plt.savefig(LOGDIR+'/best5.png')

    # show the first few worst images
    n_samples = 5
    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(2, n_samples, index + 1)
        sample_image = dataset[final_losses_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Rank: ' + str(2**14-index))
        plt.subplot(2, n_samples, n_samples + index + 1)
        sample_image = final_reconstructions[final_losses_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[-(index+1)]])))
    plt.savefig(LOGDIR+'/worst5.png')

    run_ID += 1

# save final results (a matrix with the order of best configurations for each network type - for example if a row is
# [2 0 1], it means that network 2 had the lowest loss, then net 0 and finally net 1). Analysis in analyse_results.py.
if not os.path.exists('./results'):
    os.mkdir('./results')
np.save('./results' + model_type + '_final_losses_order_all', final_losses_order_all)
