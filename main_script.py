from batchMaker import StimMaker
from parameters import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from create_sprite import images_to_sprite, invert_grayscale
import itertools
sys.setrecursionlimit(1500)  # need to make the recursion limit higher

########################################################################################################################
# Create dataset
########################################################################################################################

stim_maker = StimMaker(im_size, shape_size, bar_width) # handles data generation
if not os.path.exists('./dataset.npy'):
    flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
    matrices = np.reshape(flat_matrices, [-1, 3, 5])
    dataset = np.zeros(shape=(matrices.shape[0], im_size[0], im_size[1], 1))  # need fourth dimension for tensorflow

    for i in range(2**15):
        dataset[i, :, :, :], _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :]*(other_shape_ID-1) + 1, doVernier=False)
        print("\rMaking dataset: {}/{} ({:.1f}%)".format(i, 2**15, i * 100 / 2**15),end="")

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
        sample_image = dataset[np.random.randint(2**15), :, :].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
    plt.show()

# we do a loop over many diffent number of hidden units
final_losses_order_all = np.zeros(shape=(n_hidden_units_max, 2**15))

for n_hidden_units in range(1, n_hidden_units_max+1):

    print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
    ### LOGDIR  ###
    LOGDIR = './' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
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
        with tf.name_scope('auto_encoder'):
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
    elif model_type is 'conv':
        with tf.name_scope('auto_encoder'):
            with tf.name_scope('neurons'):
                conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
                tf.summary.histogram('conv1', conv1)
                conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
                tf.summary.histogram('conv2', conv2)
                dense = tf.layers.dense(conv2, n_hidden_units, name='hidden_layer')
                tf.summary.histogram('dense', dense)
                X_reconstructed = tf.layers.dense(dense, im_size[0]*im_size[1], name='reconstruction')
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


    ########################################################################################################################
    # Create embedding of the secondary capsules output
    ########################################################################################################################
    #
    # with tf.device('/cpu:0'):
    #
    #     embedding_writer = tf.summary.FileWriter(LOGDIR)  # to write summaries
    #     np.random.shuffle(dataset)
    #     embedding_data = dataset[:5000, :, :, :]  # we will pass the entire dataset to the embeddings
    #
    #     # create sprites (if they don't exist yet)
    #     if not os.path.exists(LOGDIR+'/sprites.png'):
    #         sprites = invert_grayscale(images_to_sprite(np.squeeze(embedding_data)))
    #         plt.imsave(LOGDIR+'/sprites.png', sprites, cmap='gray')
    #     SPRITES = LOGDIR+'/sprites.png'
    #
    #     # embeddings for the input images
    #     embedding_input_images = tf.reshape(X_flat, [-1, im_size[0]*im_size[1]])
    #     embedding_size_images = im_size[0]*im_size[1]
    #     embedding_images = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_images]), name='input_images_embedding')
    #     assignment_images = embedding_images.assign(embedding_input_images)
    #     # embeddings for the hidden layer activations
    #     embedding_input_hidden = tf.reshape(hidden, [-1, n_hidden_units])
    #     embedding_size_hidden = n_hidden_units
    #     embedding_hidden = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_hidden]), name='hidden_layer_embedding')
    #     assignment_hidden = embedding_hidden.assign(embedding_input_hidden)
    #     # embeddings for the reconstructed images
    #     embedding_input_reconstructions = tf.reshape(X_reconstructed, [-1, im_size[0]*im_size[1]])
    #     embedding_size_reconstructions = im_size[0] * im_size[1]
    #     embedding_reconstructions = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_reconstructions]), name='reconstructed_images_embedding')
    #     assignment_reconstructions = embedding_reconstructions.assign(embedding_input_reconstructions)
    #
    #     # configure embedding visualizer
    #     # input images embedding
    #     config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    #     # embedding_config_images = config.embeddings.add()
    #     # embedding_config_images.tensor_name = embedding_images.name
    #     # embedding_config_images.sprite.image_path = SPRITES
    #     # embedding_config_images.sprite.single_image_dim.extend([max(im_size), max(im_size)])
    #     # hidden layer embedding
    #     embedding_config_hidden = config.embeddings.add()
    #     embedding_config_hidden.tensor_name = embedding_hidden.name
    #     embedding_config_hidden.sprite.image_path = SPRITES
    #     embedding_config_hidden.sprite.single_image_dim.extend([max(im_size), max(im_size)])
        # reconstructed images embedding
        # embedding_config_reconstructions = config.embeddings.add()
        # embedding_config_reconstructions.tensor_name = embedding_reconstructions.name
        # embedding_config_reconstructions.sprite.image_path = SPRITES
        # embedding_config_reconstructions.sprite.single_image_dim.extend([max(im_size), max(im_size)])



    ########################################################################################################################
    # Training
    ########################################################################################################################

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
    n_batches = 2**15//batch_size

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

                # shuffle data
                np.random.shuffle(dataset)
                for batch in range(n_batches):
                    # get new batch
                    batch_data = dataset[batch:batch+batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset[batch:batch+batch_size, :, :, :].shape)

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

        # get all post-training losses
        final_losses,  final_reconstructions = sess.run([all_losses, X_reconstructed_image], feed_dict={X: dataset+ np.random.normal(0, late_noise, size=dataset.shape)})

    # get indices of the configurations from lowest to highest loss
    final_losses_order = final_losses.argsort()
    final_losses_order_all[n_hidden_units-1, :] = final_losses_order  # the -1 because the loop goes from 1->n_hidden_units_max, but indexing starts at 0

    # show the first few best images
    n_samples = 5
    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(2, n_samples, index + 1)
        sample_image = dataset[final_losses_order[index], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Configuration - rank (1=BEST): ' + str(index))
        plt.subplot(2, n_samples, n_samples + index + 1)
        sample_image = final_reconstructions[final_losses_order[index], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Reconstruction - rank (1=BEST): ' + str(index))
    plt.savefig(LOGDIR+'/best5.png')

    # show the first few worst images
    n_samples = 5
    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(2, n_samples, index + 1)
        sample_image = dataset[final_losses_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Configuration - rank: (1=WORST): ' + str(index))
        plt.subplot(2, n_samples, n_samples + index + 1)
        sample_image = final_reconstructions[final_losses_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
        plt.title('Reconstruction - rank: (1=WORST): ' + str(index))
    plt.savefig(LOGDIR+'/worst5.png')

# save final results (a matrix with the order of best configurations for each network type) and plot it
np.save('final_losses_order_all', final_losses_order_all)
