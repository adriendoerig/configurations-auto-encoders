### set to 1 if running in the cloud (will change data/saving path names accordingly) ###
in_cloud = 0
do_training = 0
do_analysis = True


# see ae_model_fn.py for model details
# models = ['dense', 'large_dense', 'conv', 'large_conv', 'caps', 'caps_16_dims', 'large_caps', 'large_caps_16_dims', 'VAE', 'VAE_beta2', 'VAE_conv', 'VAE_conv_beta2', 'alexnet_layers_1_3', 'alexnet_layers_1_5']
models = ['caps_16_dims', 'large_caps_16_dims']
# models = ['alexnet_layers_1_3', 'alexnet_layers_1_5']

### stimulus params ###
im_size = (32, 52)                             # size of full image
other_shape_ID = 10                              # there will be squares and this shape in the array
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
    tfrecords_path_test = 'gs://autoencoders-data/dataset_test_imsz_'+str(im_size[0])+str(im_size[1])+'.npy'
    results_folder = 'gs://autoencoders-data/results_'+str(im_size[0])+str(im_size[1])
else:
    tfrecords_path_train = './dataset_train_imsz_'+str(im_size[0])+str(im_size[1])+'.tfrecords'
    npy_dataset_path_test = './dataset_test_imsz_'+str(im_size[0])+str(im_size[1])+'.npy'
    results_folder = './results_'+str(im_size[0])+str(im_size[1])

learning_rate = .00005
batch_size = 64
n_epochs = 10
n_steps = 2 ** 15 // batch_size * n_epochs
eval_steps = 10  # number of of steps for which to evaluate model
eval_throttle_secs = 100000  # number of seconds after which to evaluate model

