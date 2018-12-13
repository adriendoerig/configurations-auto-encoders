import logging
import tensorflow as tf
from ae_model_fn import model_fn
from ae_input_fn import input_fn

print('################################################################################################')
print('CAREFUL!')
print('I REPEAT: CAREFUL!')
print('SET USE_THESE_PARAMS=False IN PARAMETERS.PY')
print('################################################################################################')


### set to 1 if running in the cloud (will change data/saving path names accordingly) ###
in_cloud = 0
do_training = True
do_analysis = True
models = ['dense', 'large_dense', 'conv', 'large_conv', 'caps', 'large_caps', 'VAE', 'VAE_conv', 'alexnet_layer_1_3', 'alexnet_layer_1_5']

### stimulus params ###
im_size = (32, 52)                             # size of full image
other_shape_ID = 7                              # there will be squares and this shape in the array
shape_size = 10                                 # size of a single shape in pixels
random_size = False                             # shape_size will vary around shape_size
random_pixels = 0                               # stimulus pixels are drawn from random.uniform(1-random_pixels,1+random_pixels). So use 0 for deterministic stimuli. see batchMaker.py
simultaneous_shapes = 1                         # number of different shapes in an image. NOTE: more than 2 is not supported at the moment
bar_width = 1                                   # thickness of elements' bars
noise_level = 0.05                               # add noise in dataset
late_noise = 0.0                               # add noise to each batch
shape_types = [0, 1, 2, 3, 4, 5, 6, 9]          # see batchMaker.drawShape for number-shape correspondences
group_last_shapes = 1                           # attributes the same label to the last n shapeTypes
fixed_stim_position = (1,1)                     # put top left corner of all stimuli at fixed_position
normalize_images = False                        # make each image mean=0, std=1
vernier_normalization_exp = 0                   # to give more importance to the vernier (see batchMaker). Use 0 for no effect. > 0  -> favour vernier during training
normalize_sets = False                          # compute mean and std over 100 images and use this estimate to normalize each image
max_rows, max_cols = 3, 5                       # max number of rows, columns of shape grids
vernier_grids = False                           # if true, verniers come in grids like other shapes. Only single verniers otherwise.


### training params ###
if in_cloud:
    tfrecords_path_train = 'gs://autoencoders-data/dataset_train_imsz_'+str(im_size[0])+str(im_size[1])+'.tfrecords'
    tfrecords_path_test = 'gs://autoencoders-data/dataset_test_imsz_'+str(im_size[0])+str(im_size[1])+'.tfrecords'
else:
    tfrecords_path_train = './dataset_train_imsz_'+str(im_size[0])+str(im_size[1])+'.tfrecords'
    tfrecords_path_test = './dataset_test_imsz_'+str(im_size[0])+str(im_size[1])+'.tfrecords'

learning_rate = .00005
batch_size = 64
n_epochs = 10
n_steps = 2 ** 15 // batch_size * n_epochs
eval_steps = 10  # number of of steps for which to evaluate model
eval_throttle_secs = 100000  # number of seconds after which to evaluate model

### network params ###
# 'dense' = single dense hidden layer.
# 'dense_large' = two fc encoder layers, one bottleneck layer and two fc decoder layers
# 'conv' = two conv layers followed by a dense layer.
# 'caps' = a conv layer, a primary caps layer and a secondary caps layer.
# 'conv_large' = 3 conv laywer + pooling, then a dense bottleneck, then 3 upscaling layer as a decoder (no real deconv)

if do_training:
    for model_type in models:

        if model_type is 'dense':
            n_hidden_units_max = 128
        elif model_type is 'large_dense':
            n_neurons1 = 50
            n_neurons2 = 50
            n_hidden_units_max = 128
        elif model_type is 'conv':
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
            n_hidden_units_max = 128
        elif model_type is 'large_conv':
            # cf https://github.com/mchablani/deep-learning/blob/master/autoencoder/Convolutional_Autoencoder.ipynb
            bottleneck_features_max = 8
        elif model_type is 'caps' or model_type is 'large_caps':
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
            n_hidden_units_max = 10  # number of secondary capsules (note: 16*8=128 = Nbr of neurons than the convnet)
            caps2_n_dims = 8  # of n dimensions
            rba_rounds = 3
            if model_type is 'large_caps':
                n_neurons1 = 50
                n_neurons2 = 50
        elif model_type is 'VAE' or 'VAE_conv':
            # cf. https://danijar.com/building-variational-auto-encoders-in-tensorflow/
            # and https://colab.research.google.com/drive/1Wl78KHPzQ2Q253Rob5W1o0bd8nu9DZel#scrollTo=zaCO7S0-_KNn&forceEdit=true&offline=true&sandboxMode=true
            # good explanation https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
            if model_type is 'VAE':
                n_neurons1 = 50
                n_neurons2 = 50
            n_hidden_units_max = 128
            beta = 1  # 1 -> no disentaglement >1 -> disentanglement
        elif model_type is 'alexnet_layers_1_3' or model_type is 'alexnet_layers_1_5':
            n_hidden_units_max = 512
            zoom = 3


        print('-------------------------------------------------------')
        print('TF version:', tf.__version__)
        print('Starting encoder script...')
        print('-------------------------------------------------------')


        ###########################
        #      Preparations:      #
        ###########################
        # For reproducibility:
        tf.reset_default_graph()

        # Output the loss in the terminal every few steps:
        logging.getLogger().setLevel(logging.INFO)


        ##################################
        #    Training:                   #
        ##################################


        # we will do a loop over many diffent number of hidden units
        if model_type is 'caps' or model_type is 'large_caps':
            # we don't use ALL n_hidden_units. Here, choose which ones to use.
            chosen_n_units = range(1, n_hidden_units_max + 1)
        elif model_type is 'large_conv':
            chosen_n_units = range(1, bottleneck_features_max + 1)
        elif model_type is 'alexnet_layers_1_3' or model_type is 'alexnet_layers_1_5':
            chosen_n_units = range(16, n_hidden_units_max + 1, 16)
        else:
            chosen_n_units = range(4, n_hidden_units_max + 1, 4)

        for n_hidden_units in chosen_n_units:

            print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
            ### LOGDIR  ###
            if in_cloud:
                LOGDIR = 'gs://autoencoders-data/' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
            else:
                LOGDIR = './' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'

            # Create the estimator:
            ae = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': n_hidden_units, 'LOGDIR': LOGDIR}, model_dir=LOGDIR)
            train_spec = tf.estimator.TrainSpec(input_fn, max_steps=n_steps)
            eval_spec = tf.estimator.EvalSpec(input_fn, steps=eval_steps, throttle_secs=eval_throttle_secs)

            # Lets go!
            tf.estimator.train_and_evaluate(ae, train_spec, eval_spec)




if do_analysis:
    import os, itertools, imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from batchMaker import StimMaker
    tf.estimator.Estimator._validate_features_in_predict_input = lambda *args: None

    for model_type in models:

        if model_type is 'dense':
            n_hidden_units_max = 128
        elif model_type is 'large_dense':
            n_neurons1 = 50
            n_neurons2 = 50
            n_hidden_units_max = 128
        elif model_type is 'conv':
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
            n_hidden_units_max = 128
        elif model_type is 'large_conv':
            # cf https://github.com/mchablani/deep-learning/blob/master/autoencoder/Convolutional_Autoencoder.ipynb
            bottleneck_features_max = 8
        elif model_type is 'caps' or model_type is 'large_caps':
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
            n_hidden_units_max = 10  # number of secondary capsules (note: 16*8=128 = Nbr of neurons than the convnet)
            caps2_n_dims = 8  # of n dimensions
            rba_rounds = 3
            if model_type is 'large_caps':
                n_neurons1 = 50
                n_neurons2 = 50
        elif model_type is 'VAE' or 'VAE_conv':
            # cf. https://danijar.com/building-variational-auto-encoders-in-tensorflow/
            # and https://colab.research.google.com/drive/1Wl78KHPzQ2Q253Rob5W1o0bd8nu9DZel#scrollTo=zaCO7S0-_KNn&forceEdit=true&offline=true&sandboxMode=true
            # good explanation https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
            if model_type is 'VAE':
                n_neurons1 = 50
                n_neurons2 = 50
            n_hidden_units_max = 128
            beta = 1  # 1 -> no disentaglement >1 -> disentanglement
        elif model_type is 'alexnet_layers_1_3' or model_type is 'alexnet_layers_1_5':
            n_hidden_units_max = 512
            zoom = 3

        ########################################################################################################################
        # Make or load dataset_test
        ########################################################################################################################


        stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
        n_matrices = 2 ** 14

        if not os.path.exists('./dataset_test.npy'):
            flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
            matrices = np.reshape(flat_matrices, [-1, 3, 5])
            flat_matrices[:, 7] = 0
            unique_flat_matrices = np.unique(flat_matrices, axis=0)
            matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
            matrices[:, 1, 2] = 0

            dataset_test = np.zeros(
                shape=(n_matrices, im_size[0], im_size[1], 1))  # need fourth dimension for tensorflow

            for i in range(n_matrices):
                dataset_test[i, :, :, :], _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :] * (
                other_shape_ID - 1) + 1, doVernier=False)
                print("\rMaking dataset_test: {}/{} ({:.1f}%)".format(i, n_matrices, i * 100 / n_matrices), end="")

            np.save('dataset_test.npy', dataset_test)

        else:
            print(' dataset_test.npy found -> loading')
            dataset_test = np.load('./dataset_test.npy')


        # the following function makes tf.dataset_tests from numpy batches
        def input_fn_pred(batch):
            batch_size = batch.shape[0]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            dataset_test = tf.data.Dataset.from_tensor_slices(batch)
            dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
            # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
            dataset_test = dataset_test.prefetch(2)
            # Create an iterator for the dataset_test and the above modifications.
            iterator = dataset_test.make_one_shot_iterator()
            # Get the next batch of images and labels.
            images = iterator.get_next()
            feed_dict = {'images': images}
            return feed_dict


        # we will do a loop over many diffent number of hidden units
        if model_type is 'caps' or model_type is 'large_caps':  # we don't use ALL n_hidden_units. Here, choose which ones to use.
            chosen_n_units = range(1, n_hidden_units_max + 1)
        elif model_type is 'large_conv':
            chosen_n_units = range(1, bottleneck_features_max + 1)
        elif model_type is 'alexnet_layers_1_3' or model_type is 'alexnet_layers_1_5':
            chosen_n_units = range(16, n_hidden_units_max + 1, 16)
        else:
            chosen_n_units = range(8, n_hidden_units_max + 1, 4)

        ########################################################################################################################
        # Classify the reconstructed images from best to worst
        ########################################################################################################################


        if not os._exists('./results/' + model_type + '_final_losses_order_all.npy'):

            final_losses_order_all = np.zeros(shape=(len(chosen_n_units), n_matrices))

            for it, n_hidden_units in enumerate(chosen_n_units):

                print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
                ### LOGDIR  ###
                if in_cloud:
                    LOGDIR = 'gs://autoencoders-data/' + model_type + '/' + model_type + '_' + str(
                        n_hidden_units) + '_hidden_units_logdir'
                else:
                    LOGDIR = './' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'

                # Create the estimator:
                ae = tf.estimator.Estimator(model_fn=model_fn,
                                            params={'bottleneck_units': n_hidden_units, 'LOGDIR': LOGDIR},
                                            model_dir=LOGDIR)

                # Get losses and reconstructed images for each stimulus
                n_trials = 1
                n_batches = n_matrices // batch_size
                final_losses = np.zeros(shape=(n_trials, n_matrices))
                final_reconstructions = np.zeros(shape=(n_matrices, im_size[0], im_size[1], 1))
                for batch in range(n_batches):
                    print("\r..... {}/{} ({:.1f}%)".format(batch, n_batches, batch * 100 / n_batches), end="")
                    for trial in range(n_trials):
                        this_batch = dataset_test[batch * batch_size:batch * batch_size + batch_size, :, :,
                                     :] + np.random.normal(0, late_noise, size=dataset_test[
                                                                               batch * batch_size:batch * batch_size + batch_size,
                                                                               :, :, :].shape)
                        ae_out = list(ae.predict(input_fn=lambda: input_fn_pred(this_batch)))
                        final_losses[trial, batch * batch_size:batch * batch_size + batch_size] = [p["all_losses"] for p
                                                                                                   in ae_out]
                    final_reconstructions[batch * batch_size:batch * batch_size + batch_size, :, :, :] = [
                        p["reconstructions"] for p in ae_out]
                final_losses = np.mean(final_losses, axis=0)

                # get indices of the configurations from lowest to highest loss
                final_losses_order = final_losses.argsort()
                final_losses_order_all[it, :] = final_losses_order

                # show the first few best images
                n_samples = 5
                plt.figure(figsize=(n_samples * 2, 3))
                for index in range(n_samples):
                    plt.subplot(2, n_samples, index + 1)
                    sample_image = dataset_test[final_losses_order[index], :, :, 0].reshape(im_size[0], im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title(('Rank: ' + str(index)))
                    plt.subplot(2, n_samples, n_samples + index + 1)
                    sample_image = final_reconstructions[final_losses_order[index], :, :, 0].reshape(im_size[0],
                                                                                                     im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[index]])))
                plt.savefig(LOGDIR + '/best5.png')

                # show the first few worst images
                n_samples = 5
                plt.figure(figsize=(n_samples * 2, 3))
                for index in range(n_samples):
                    plt.subplot(2, n_samples, index + 1)
                    sample_image = dataset_test[final_losses_order[-(index + 1)], :, :, 0].reshape(im_size[0],
                                                                                                   im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Rank: ' + str(n_matrices - index))
                    plt.subplot(2, n_samples, n_samples + index + 1)
                    sample_image = final_reconstructions[final_losses_order[-(index + 1)], :, :, 0].reshape(im_size[0],
                                                                                                            im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[-(index + 1)]])))
                plt.savefig(LOGDIR + '/worst5.png')

                plt.close('all')

                # save final results (a matrix with the order of best configurations for each network type - for example if a row is
                # [2 0 1], it means that network 2 had the lowest loss, then net 0 and finally net 1). Analysis in analyse_results.py.
            if not os.path.exists('./results'):
                os.mkdir('./results')
            np.save('./results/' + model_type + '_final_losses_order_all', final_losses_order_all)

        else:
            final_losses_order_all = np.load('./results/' + model_type + '_final_losses_order_all')

        ########################################################################################################################
        # Make plots and gifs
        ########################################################################################################################


        # scores count the avg. position of a configuration: if it has the lowest loss->n_matrices, if is has the highest loss->1.
        # note that the higher the score, the better the performance (this is easier for visualizing the result graphs).
        # each line i corresponds to the mean score over all models with n_hidden_units <= i+1 (e.g. line 0 contains the scores
        # for the net with a single hidden unit and line 2 contains the avg scores over models with 1, 2 & 3 hidden units.
        print('computing mean scores over models with increasing n_hidden_units')
        scores = np.zeros(shape=(final_losses_order_all.shape[0], n_matrices))
        for i in range(final_losses_order_all.shape[0]):
            for j in range(n_matrices):
                scores[i:, j] += np.squeeze(
                    np.tile(np.where(j == final_losses_order_all[i, :]), final_losses_order_all.shape[0] - i))
            scores[i, :] /= i + 1
        scores = n_matrices + 1 - scores  # originally, the best configs have low values. Switch this for better visualisation.

        print('creating graph for final results.')
        mean_score = np.mean(final_losses_order_all, axis=0)

        ind = np.arange(n_matrices)
        fig, ax = plt.subplots()
        ax.bar(ind, mean_score, color=(3. / 255, 57. / 255, 108. / 255))

        # add some text for labels, title and axes ticks, and save figure
        ax.set_xlabel('configuration IDs')
        ax.set_ylabel('Mean scores')
        plt.title('Current mean over networks 1 -> ' + str(chosen_n_units[-1]))
        plt.savefig('./results/' + model_type + '_mean_scores.png')

        # plot five best and five worst configs
        mean_score_order = mean_score.argsort()
        n_samples = 5
        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(2, n_samples, index + 1)
            sample_image = dataset_test[mean_score_order[index], :, :, 0].reshape(im_size[0], im_size[1])
            plt.imshow(sample_image, cmap="binary")
            plt.axis("off")
            plt.title('Best configs - rank: (1=BEST): ' + str(index))
            plt.subplot(2, n_samples, n_samples + index + 1)
            sample_image = dataset_test[mean_score_order[-(index + 1)], :, :, 0].reshape(im_size[0], im_size[1])
            plt.imshow(sample_image, cmap="binary")
            plt.axis("off")
            plt.title('Worst configs - rank: (1=WORST): ' + str(index))
        plt.savefig('./results/' + model_type + '_mean_scores_best_and_worst_configs.png')

        # make a cool gif showing the evolution of mean_score as neurons are added to the hidden layer
        print('creating gif of results across networks')
        imgs_for_gif = []


        def plot_for_offset(data):
            plt.close('all')
            mean_score = np.mean(data, axis=0)
            ind = np.arange(n_matrices)
            fig, ax = plt.subplots()
            ax.bar(ind, mean_score, color=(3. / 255, 57. / 255, 108. / 255))
            ax.set_xlabel('configuration IDs')
            ax.set_ylabel('Mean scores')
            ax.set_ylim(0, n_matrices)
            plt.title('Current mean over networks 1 -> ' + str(data.shape[0]))
            # Used to return the plot as an image array
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image


        # make gif
        for i in range(1, len(chosen_n_units) + 1):
            print("\r{}/{} ({:.1f}%) ".format(i, len(chosen_n_units), i * 100 / len(chosen_n_units)), end="")
            imgs_for_gif.append(plot_for_offset(final_losses_order_all[:i, :]))
        imageio.mimsave('./results/' + model_type + '_mean_scores_evolving.gif', imgs_for_gif, fps=2)
