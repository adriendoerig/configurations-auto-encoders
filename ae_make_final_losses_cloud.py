# -*- coding: utf-8 -*-
"""
Goes through all trained models of type model_type (set in parameters.py)
-> saves model_type_final_losses_order_all.npy
"""

import numpy as np
from parameters import *
from ae_model_fn import model_fn


########################################################################################################################
# Make or load dataset
########################################################################################################################

dataset = np.load('gs://autoencoders-data/dataset.npy')

def input_fn_pred(batch):
    return {'images': tf.convert_to_tensor(batch)}


########################################################################################################################
# Classify the reconstructed images from best to worst
########################################################################################################################


# we will do a loop over many diffent number of hidden units
if model_type is ('caps' or 'large_caps'):  # we don't use ALL n_hidden_units. Here, choose which ones to use.
    chosen_n_units = range(1, n_hidden_units_max + 1)
elif model_type is 'large_conv':
    chosen_n_units = range(1, bottleneck_features_max + 1)
else:
    chosen_n_units = range(8, n_hidden_units_max + 1, 4)

final_losses_order_all = np.zeros(shape=(len(chosen_n_units), 2**15))

for it, n_hidden_units in enumerate(chosen_n_units):

    print('\rCurrent model: ' + model_type + ' with ' + str(n_hidden_units) + ' hidden units.')
    ### LOGDIR  ###
    if in_cloud:
        LOGDIR = 'gs://autoencoders-data/' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
        checkpoint_path = LOGDIR + '/checkpoint.ckpt'
    else:
        LOGDIR = './' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'
        checkpoint_path = LOGDIR + '/checkpoint.ckpt'

    # Create the estimator:
    ae = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': n_hidden_units, 'LOGDIR': LOGDIR}, model_dir=checkpoint_path)

    # Get losses and reconstructed images for each stimulus
    n_trials = 10
    n_batches = 2 ** 15 // batch_size
    final_losses = np.zeros(shape=(n_trials, 2**15))
    final_reconstructions = np.zeros(shape=(2**15, im_size[0], im_size[1], 1))
    for batch in range(n_batches):
        for trial in range(n_trials):
            this_batch = dataset[batch * batch_size:batch * batch_size + batch_size, :, :, :] + np.random.normal(0, late_noise, size=dataset[batch * batch_size:batch * batch_size + batch_size, :, :, :].shape)
            ae_out = list(ae.predict(input_fn=lambda: input_fn_pred(this_batch), checkpoint_path=checkpoint_path))
            final_losses[trial, batch * batch_size:batch * batch_size + batch_size] = [p["all_losses"] for p in ae_out]
        final_reconstructions[batch * batch_size:batch * batch_size + batch_size, :, :, :] = [p["X_reconstructed"] for p in ae_out]
    final_losses = np.mean(final_losses, axis=0)

    # get indices of the configurations from lowest to highest loss
    final_losses_order = final_losses.argsort()
    final_losses_order_all[it, :] = final_losses_order

    # save final results (a matrix with the order of best configurations for each network type - for example if a row is
    # [2 0 1], it means that network 2 had the lowest loss, then net 0 and finally net 1). Analysis in analyse_results.py.
    np.save('gs://autoencoders-data/results/' + model_type + '_final_losses_order_all', final_losses_order_all)