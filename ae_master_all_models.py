import logging, os
import tensorflow as tf
from ae_model_fn import model_fn
from ae_input_fn import input_fn, input_fn_pred
from ae_make_tfrecords import make_tfrecords
from ae_master_all_models_params import *


print('################################################################################################')
print('CAREFUL!')
print('I REPEAT: CAREFUL!')
print('SET USE_THESE_PARAMS=False IN PARAMETERS.PY')
print('################################################################################################')

if do_training:

    if not os.path.exists(tfrecords_path_train):
        make_tfrecords('train')

    for model_type in models:

        # we will do a loop over many diffent number of hidden units
        if 'caps' in model_type:
            if '16_dims' in model_type:
                n_hidden_units_max = 4
            else:
                n_hidden_units_max = 16  # number of secondary capsules (note: 16*4=64 = Nbr of neurons in other nets)
            chosen_n_units =  range(1, n_hidden_units_max + 1)
        elif 'alexnet' in model_type:
            n_hidden_units_max = 32
            chosen_n_units = range(2, n_hidden_units_max + 1, 2)
        else:
            n_hidden_units_max = 64
            chosen_n_units = range(4, n_hidden_units_max + 1, 4)

        print('-------------------------------------------------------')
        print('TF version:', tf.__version__)
        print('Starting autoencoder script...')
        print('-------------------------------------------------------')

        # For reproducibility:
        tf.reset_default_graph()

        # Output the loss in the terminal every few steps:
        logging.getLogger().setLevel(logging.INFO)

        for n_hidden_units in chosen_n_units:

            print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
            ### LOGDIR  ###
            if in_cloud:
                LOGDIR = 'gs://autoencoders-data/' + model_type + '_imsz_'+str(im_size[0])+str(im_size[1])+ '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
            else:
                LOGDIR = './' + model_type + '_imsz_'+str(im_size[0])+str(im_size[1])+ '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'

            # Create the estimator:
            ae = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': n_hidden_units, 'LOGDIR': LOGDIR, 'model_type': model_type}, model_dir=LOGDIR)
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

    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    for model_type in models:

        # we will do a loop over many diffent number of hidden units
        if 'caps' in model_type:
            if '16_dims' in model_type:
                n_hidden_units_max = 4
            else:
                n_hidden_units_max = 16  # number of secondary capsules (note: 16*4=64 = Nbr of neurons in other nets)
            chosen_n_units = range(1, n_hidden_units_max + 1)
        elif 'alexnet' in model_type:
            n_hidden_units_max = 32
            chosen_n_units = range(2, n_hidden_units_max + 1, 2)
        else:
            n_hidden_units_max = 64
            chosen_n_units = range(4, n_hidden_units_max + 1, 4)

        ########################################################################################################################
        # Make or load dataset_test
        ########################################################################################################################


        stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
        n_matrices = 2 ** 14

        if not os.path.exists(npy_dataset_path_test):
            flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
            matrices = np.reshape(flat_matrices, [-1, 3, 5])
            flat_matrices[:, 7] = 0
            unique_flat_matrices = np.unique(flat_matrices, axis=0)
            matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
            matrices[:, 1, 2] = 0

            dataset_test = np.zeros(shape=(n_matrices, im_size[0], im_size[1], 1))  # need fourth dimension for tensorflow

            for i in range(n_matrices):
                dataset_test[i, :, :, :], _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :] * (
                other_shape_ID - 1) + 1, doVernier=False)
                print("\rMaking dataset_test: {}/{} ({:.1f}%)".format(i, n_matrices, i * 100 / n_matrices), end="")

            np.save(npy_dataset_path_test, dataset_test)

        else:
            print(' dataset_test.npy found -> loading')
            dataset_test = np.load(npy_dataset_path_test)


        ########################################################################################################################
        # Classify the reconstructed images from best to worst
        ########################################################################################################################


        if not os.path.exists(results_folder + '/' + model_type + '_final_losses_order_all.npy'):

            final_losses_order_all = np.zeros(shape=(len(chosen_n_units), n_matrices))

            for it, n_hidden_units in enumerate(chosen_n_units):

                print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
                ### LOGDIR  ###
                if in_cloud:
                    LOGDIR = 'gs://autoencoders-data/' + model_type + '_imsz_'+str(im_size[0])+str(im_size[1]) + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
                else:
                    LOGDIR = './' + model_type + '_imsz_'+str(im_size[0])+str(im_size[1]) + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'

                # Create the estimator:
                ae = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': n_hidden_units, 'LOGDIR': LOGDIR, 'model_type': model_type}, model_dir=LOGDIR)

                # Get losses and reconstructed images for each stimulus
                n_trials = 1
                n_batches = n_matrices // batch_size
                final_losses = np.zeros(shape=(n_trials, n_matrices))

                for batch in range(n_batches):
                    print("\r..... {}/{} ({:.1f}%)".format(batch, n_batches, batch * 100 / n_batches), end="")
                    for trial in range(n_trials):
                        this_batch = dataset_test[batch * batch_size:batch * batch_size + batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset_test[ batch * batch_size:batch * batch_size + batch_size, :, :, :].shape)
                        ae_out = list(ae.predict(input_fn=lambda: input_fn_pred(this_batch)))
                        final_losses[trial, batch * batch_size:batch * batch_size + batch_size] = [p["all_losses"] for p in ae_out]
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
                    ae_out = list(ae.predict(input_fn=lambda: input_fn_pred(tf.expand_dims(tf.cast(dataset_test[final_losses_order[index], :, :, :], tf.float32), axis=0), return_batch_size=True)))
                    img = [p["reconstructions"] for p in ae_out]
                    img = np.array(img)
                    if 'alexnet' in model_type:
                        sample_image = img[0, :, :, :].reshape(227, 227, 3)
                    else:
                        sample_image = img[0, :, :, :].reshape(im_size[0], im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[index]])))
                plt.savefig(LOGDIR + '/best5.png')

                # show the first few worst images
                n_samples = 5
                plt.figure(figsize=(n_samples * 2, 3))
                for index in range(n_samples):
                    plt.subplot(2, n_samples, index + 1)
                    sample_image = dataset_test[final_losses_order[-(index + 1)], :, :, 0].reshape(im_size[0], im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Rank: ' + str(n_matrices - index))
                    plt.subplot(2, n_samples, n_samples + index + 1)
                    ae_out = list(ae.predict(input_fn=lambda: input_fn_pred(tf.expand_dims(tf.cast(dataset_test[final_losses_order[-(index+1)], :, :, :], tf.float32), axis=0), return_batch_size=True)))
                    img = [p["reconstructions"] for p in ae_out]
                    img = np.array(img)
                    if 'alexnet' in model_type:
                        sample_image = img[0, :, :, :].reshape(227, 227, 3)
                    else:
                        sample_image = img[0, :, :, :].reshape(im_size[0], im_size[1])
                    plt.imshow(sample_image, cmap="binary")
                    plt.axis("off")
                    plt.title('Avg. loss: ' + str(int(final_losses[final_losses_order[-(index + 1)]])))
                plt.savefig(LOGDIR + '/worst5.png')

                plt.close('all')

                # save final results (a matrix with the order of best configurations for each network type - for example if a row is
                # [2 0 1], it means that network 2 had the lowest loss, then net 0 and finally net 1). Analysis in analyse_results.py.
            if not os.path.exists(results_folder):
                os.mkdir(results_folder)
            np.save(results_folder + '/' + model_type + '_final_losses_order_all', final_losses_order_all)

        else:
            final_losses_order_all = np.load(results_folder + '/' + model_type + '_final_losses_order_all.npy')

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
                scores[i:, j] += np.squeeze(np.tile(np.where(j == final_losses_order_all[i, :]), final_losses_order_all.shape[0] - i))
            scores[i, :] /= i + 1
        scores = n_matrices - scores  # originally, the best configs have low values. Switch this for better visualisation.

        print('creating graph for final results.')

        ind = np.arange(n_matrices)
        fig, ax = plt.subplots()
        ax.bar(ind, scores[-1, :], color=(3. / 255, 57. / 255, 108. / 255))

        # add some text for labels, title and axes ticks, and save figure
        ax.set_xlabel('configuration IDs')
        ax.set_ylabel('Mean scores')
        plt.title('Current mean over networks 1 -> ' + str(chosen_n_units[-1]))
        plt.savefig(results_folder + '/' + model_type + '_mean_scores.png')

        # plot five best and five worst configs
        mean_score_order = scores[-1, :].argsort()
        n_samples = 5
        plt.figure(figsize=(n_samples * 2, 3))
        for index in range(n_samples):
            plt.subplot(2, n_samples, index + 1)
            sample_image = dataset_test[mean_score_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
            plt.imshow(sample_image, cmap="binary")
            plt.axis("off")
            plt.title('Best configs: (1=BEST): ' + str(index))
            plt.subplot(2, n_samples, n_samples + index + 1)
            sample_image = dataset_test[mean_score_order[index], :, :, 0].reshape(im_size[0], im_size[1])
            plt.imshow(sample_image, cmap="binary")
            plt.axis("off")
            plt.title('Worst configs: (1=WORST): ' + str(index))
        plt.savefig(results_folder + '/' + model_type + '_mean_scores_best_and_worst_configs.png')

        # make a cool gif showing the evolution of mean_score as neurons are added to the hidden layer
        print('creating gif of results across networks')
        imgs_for_gif = []


        def plot_for_offset(data, net_nr):
            plt.close('all')
            ind = np.arange(n_matrices)
            fig, ax = plt.subplots()
            ax.bar(ind, data, color=(3. / 255, 57. / 255, 108. / 255))
            ax.set_xlabel('configuration IDs')
            ax.set_ylabel('Mean scores')
            ax.set_ylim(0, n_matrices)
            plt.title('Current mean over networks 1 -> ' + str(net_nr))
            # Used to return the plot as an image array
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            return image


        # make gif
        for i in range(len(chosen_n_units)):
            print("\r{}/{} ({:.1f}%) ".format(i, len(chosen_n_units), i * 100 / len(chosen_n_units)), end="")
            imgs_for_gif.append(plot_for_offset(scores[i, :], i+1))
        imageio.mimsave(results_folder + '/' + model_type + '_mean_scores_evolving.gif', imgs_for_gif, fps=2)
