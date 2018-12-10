import tensorflow as tf

### set to 1 if running in the cloud (will change data/saving path names accordingly) ###
in_cloud = 1

### stimulus params ###
im_size = (50, 83)                             # size of full image
other_shape_ID = 2                              # there will be squares and this shape in the array
shape_size = 16                                 # size of a single shape in pixels
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

### network params ###
# 'dense' = single dense hidden layer.
# 'dense_large' = two fc encoder layers, one bottleneck layer and two fc decoder layers
# 'conv' = two conv layers followed by a dense layer.
# 'caps' = a conv layer, a primary caps layer and a secondary caps layer.
# 'conv_large' = 3 conv laywer + pooling, then a dense bottleneck, then 3 upscaling layer as a decoder (no real deconv)

model_type = 'caps'

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
elif model_type is ('caps' or 'large_caps'):
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
elif model_type is 'VAE':
    # cf. https://danijar.com/building-variational-auto-encoders-in-tensorflow/
    # and https://colab.research.google.com/drive/1Wl78KHPzQ2Q253Rob5W1o0bd8nu9DZel#scrollTo=zaCO7S0-_KNn&forceEdit=true&offline=true&sandboxMode=true
    n_neurons1 = 50
    n_neurons2 = 50
    n_hidden_units_max = 128
    beta = 4  # 1 -> no disentaglement >1 -> disentanglement

# learning rate
learning_rate = .0001

### training opts ###
if in_cloud:
    tfrecords_path = 'gs://autoencoders-data/dataset.tfrecords'
else:
    tfrecords_path = './dataset.tfrecords'

batch_size = 64
n_epochs = 10
n_steps = 2**15//batch_size*n_epochs
eval_steps = 10  # number of of steps for which to evaluate model
eval_throttle_secs = 100000  # number of seconds after which to evaluate model
