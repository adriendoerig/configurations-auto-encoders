# -*- coding: utf-8 -*-
"""
Runs a model of type model_type (defined in parameters.py) with a number of n_bottleneck_units and saves the models
in LOGDIR. The results are analysed in ae_make_final_losses.py and ae_analyse_results.py.
"""

import logging
import numpy as np
from parameters import *
from ae_model_fn import model_fn
from ae_input_fn import input_fn


print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('-------------------------------------------------------')


###########################
#      Preparations:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)
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
else:
    chosen_n_units = range(8, n_hidden_units_max + 1, 4)

for n_hidden_units in chosen_n_units:

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
    train_spec = tf.estimator.TrainSpec(input_fn, max_steps=n_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn, steps=eval_steps, throttle_secs=eval_throttle_secs)

    # Lets go!
    tf.estimator.train_and_evaluate(ae, train_spec, eval_spec)
