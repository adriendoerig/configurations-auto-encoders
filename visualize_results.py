from visualization_functions import visualizeReconstructedImages, computeTSNEProjectionOfPixelSpace, computeTSNEProjectionOfLatentSpace, \
    twoDimensionalTsneGrid, visualizeInterpolation, visualizeArithmetics, tensorboard_embeddings, show_n_best_and_worst_configs, make_losses_and_scores_barplot, \
    show_n_best_and_worst_configs_control, make_gif_from_frames

from ae_master_all_models_params import im_size, shape_size, bar_width, npy_dataset_path_visualization, other_shape_ID
from batchMaker import StimMaker
import tensorflow as tf
import numpy as np
import os
import itertools

# model params
model_type = 'VAE_beta2'
latent_dim = 40
n_squares_in_visualization_set = 9  # choose how many squares are in the displays. if None, all stimuli will be present
LOGDIR = './imsz_3252_diamonds/' + model_type + '_imsz_'+str(im_size[0])+str(im_size[1]) + '/' + model_type + '_' + str(latent_dim) + '_hidden_units_logdir'

print('#######################################################')
print('# CREATING VISUALIZATIONS FROM MODEL IN ' + LOGDIR + '.')
print('#######################################################')

# load or create dataset
stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation


if not os.path.exists(npy_dataset_path_visualization):

    if n_squares_in_visualization_set is None:
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
        matrices = np.reshape(flat_matrices, [-1, 3, 5])
        flat_matrices[:, 7] = 0
        unique_flat_matrices = np.unique(flat_matrices, axis=0)
        matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
        matrices[:, 1, 2] = 0
        n_matrices = 2 ** 14
    else:
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
        matrices = np.reshape(flat_matrices, [-1, 3, 5])
        flat_matrices[:, 7] = 0
        # remove entries with the wrong number of squares
        row_sums = np.sum(flat_matrices, axis=1)
        flat_matrices = flat_matrices[row_sums == 15 - n_squares_in_visualization_set, :]
        unique_flat_matrices = np.unique(flat_matrices, axis=0)
        matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
        matrices[:, 1, 2] = 0
        n_matrices = matrices.shape[0]


    dataset_visualization = np.zeros(shape=(n_matrices, im_size[0], im_size[1], 1))  # need fourth dimension for tensorflow

    for i in range(n_matrices):
        dataset_visualization[i, :, :, :], _ = stim_maker.makeConfigBatch(batchSize=1, configMatrix=matrices[i, :, :] * (other_shape_ID - 1) + 1, doVernier=False)
        print("\rMaking dataset_visualization: {}/{} ({:.1f}%)".format(i, n_matrices, i * 100 / n_matrices), end="")

    np.save(npy_dataset_path_visualization, dataset_visualization)

else:
    print(' dataset_visualization.npy found -> loading')
    dataset_visualization = np.load(npy_dataset_path_visualization)


# np.random.shuffle(dataset_visualization)
from ae_model_fn import model_fn
model = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': latent_dim, 'LOGDIR': LOGDIR, 'model_type': model_type, 'process_single_image': False}, model_dir=LOGDIR)

# visualizeReconstructedImages(dataset_visualization[:64,:,:,:], im_size, model, save=False)
# computeTSNEProjectionOfPixelSpace(dataset_visualization, im_size, display=True)
# computeTSNEProjectionOfLatentSpace(dataset_visualization, im_size, model, display=True)
# twoDimensionalTsneGrid(dataset_visualization[:int(np.floor(np.sqrt(dataset_visualization.shape[0]))**2),:,:,:], im_size, model, 1, int(np.floor(np.sqrt(dataset_visualization.shape[0]))), pixel_or_latent='latent', display=True)
# visualizeInterpolation(dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], model, im_size)
# visualizeArithmetics(dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0,dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0,dataset_visualization.shape[0])], model, im_size)
# tensorboard_embeddings(dataset_visualization, im_size, latent_dim, model, './embeddings/'+model_type+'_'+str(latent_dim)+'_latent_dims')
# show_n_best_and_worst_configs(dataset_visualization, im_size, 64, model)
# show_n_best_and_worst_configs_control(dataset_visualization, im_size, 64)
# make_losses_and_scores_barplot(dataset_visualization, model)
make_gif_from_frames(dataset_visualization, im_size, model, model_type, range(4,64+1,4), type='losses_and_scores', save_path='./')
# make_gif_from_frames(dataset_visualization, im_size, model, model_type, range(4,64+1,4), type='configs_and_loss_curves', save_path='./')