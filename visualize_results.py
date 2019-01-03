from visualization_functions import visualizeReconstructedImages, computeTSNEProjectionOfPixelSpace, computeTSNEProjectionOfLatentSpace, \
    twoDimensionalTsneGrid, visualizeInterpolation, visualizeArithmetics, tensorboard_embeddings, tensorboard_pixelspace_embedding, \
    show_n_best_and_worst_configs, make_losses_and_scores_barplot, show_n_best_and_worst_configs_control, make_gif_from_frames, \
    adjacent_squares_losses, mean_losses_and_scores_over_latent_dims
from ae_master_all_models_params import im_size, shape_size, bar_width, other_shape_ID
from batchMaker import StimMaker
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import itertools

# model params
# models = ['dense', 'large_dense', 'conv', 'large_conv', 'caps', 'caps_16_dims', 'large_caps', 'large_caps_16_dims', 'VAE', 'VAE_beta2', 'VAE_conv', 'VAE_conv_beta2', 'alexnet_layers_1_3', 'alexnet_layers_1_5']
models = ['dense']
n_squares_in_visualization_set = None  # choose how many squares are in the displays. if None, all stimuli will be present
res_path = './results_' + str(im_size[0]) + str(im_size[1])
vis_path = res_path + '/visualizations_' + str(n_squares_in_visualization_set)
npy_dataset_path_visualization = './dataset_visualization_imsz_'+str(im_size[0])+str(im_size[1])+'_'+str(n_squares_in_visualization_set)+'.npy'
npy_matrices_path_visualization = './dataset_matrices_imsz_'+str(im_size[0])+str(im_size[1])+'_'+str(n_squares_in_visualization_set)+'.npy'

# make sure all folders exist
if not os.path.exists(res_path):
    os.mkdir(res_path)
if not os.path.exists(vis_path):
    os.mkdir(vis_path)
if not os.path.exists(vis_path + '/reconstructions'):
    os.mkdir(vis_path + '/reconstructions')
if not os.path.exists(vis_path + '/latent_tsne'):
    os.mkdir(vis_path + '/latent_tsne')
if not os.path.exists(vis_path + '/tsne_grid'):
    os.mkdir(vis_path + '/tsne_grid')
if not os.path.exists(vis_path + '/interpolations'):
    os.mkdir(vis_path + '/interpolations')
if not os.path.exists(vis_path + '/arithmetics'):
    os.mkdir(vis_path + '/arithmetics')
if not os.path.exists(vis_path + '/best_worst'):
    os.mkdir(vis_path + '/best_worst')
if not os.path.exists(vis_path + '/losses_scores'):
    os.mkdir(vis_path + '/losses_scores')
if not os.path.exists(vis_path + '/adjacent_squares_losses'):
    os.mkdir(vis_path + '/adjacent_squares_losses')
if not os.path.exists(vis_path + '/losses_and_scores_mean'):
    os.mkdir(vis_path + '/losses_and_scores_mean')
if not os.path.exists(vis_path + '/gifs'):
    os.mkdir(vis_path + '/gifs')

# load or create dataset
stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation

if not os.path.exists(npy_dataset_path_visualization) or not os.path.exists(npy_matrices_path_visualization):

    if n_squares_in_visualization_set is None:
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
        flat_matrices[:, 7] = 0
        unique_flat_matrices = np.unique(flat_matrices, axis=0)
        matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
        matrices[:, 1, 2] = 0
        n_matrices = 2 ** 14
    else:
        flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
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
    np.save(npy_matrices_path_visualization, matrices)

else:
    print(' dataset_visualization.npy found -> loading')
    dataset_visualization = np.load(npy_dataset_path_visualization)
    matrices = np.load(npy_matrices_path_visualization)


for model_type in models:
    from ae_model_fn import model_fn

    # we will do a loop over many diffent number of hidden units
    if 'caps' in model_type:
        if '16_dims' in model_type:
            n_hidden_units_max = 4
        else:
            n_hidden_units_max = 10  # was 16 but it gave OOMs  # number of secondary capsules (note: 16*4=64 = Nbr of neurons in other nets)
        chosen_n_units = range(1, n_hidden_units_max + 1)
    elif 'alexnet' in model_type:
        n_hidden_units_max = 32
        chosen_n_units = range(2, n_hidden_units_max + 1, 2)
    else:
        n_hidden_units_max = 64
        chosen_n_units = range(4, n_hidden_units_max + 1, 4)


    for latent_dim in chosen_n_units:

        LOGDIR = './imsz_3252_diamonds/' + model_type + '_imsz_' + str(im_size[0]) + str(im_size[1]) + '/' + model_type + '_' + str(latent_dim) + '_hidden_units_logdir'
        # define the adequate model estimator (weights etc will be fetched from the LOGDIR)
        model = tf.estimator.Estimator(model_fn=model_fn, params={'bottleneck_units': latent_dim, 'LOGDIR': LOGDIR, 'model_type': model_type}, model_dir=LOGDIR)

        print('\n##############################################################################################################')
        print('# CREATING VISUALIZATIONS FROM MODEL IN ' + LOGDIR + '.')
        print('##############################################################################################################\n')

        if not os.path.exists(vis_path + '/__pixel_space_tsne.png'):
            computeTSNEProjectionOfPixelSpace(dataset_visualization, im_size, save_path=vis_path+'/__')
        if not os.path.exists(vis_path + '/__best_worst_configs_control.png'):
            show_n_best_and_worst_configs_control(dataset_visualization, im_size, 64, save_path=vis_path + '/__')
        if not os.path.exists(vis_path+'/embeddings/__pixelspace_embedding'):
            tensorboard_pixelspace_embedding(dataset_visualization, im_size, vis_path+'/embeddings/__pixelspace_embedding')

        plt.close('all')

        # save_path = vis_path + '/reconstructions/' + model_type + '_latentdim_' + str(latent_dim) + '_'
        # visualizeReconstructedImages(dataset_visualization[:64, :, :, :], im_size, model, model_type=model_type, save_path=save_path)
        save_path = vis_path + '/adjacent_squares_losses/' + model_type + '_latentdim_' + str(latent_dim) + '_'
        adjacent_squares_losses(dataset_visualization, matrices, model, save_path=save_path)
    #     if 'alexnet' or '16_dims' in model_type:
    #         save_path = vis_path + '/embeddings/' + model_type + '_' + str(latent_dim) + '_latent_dims'
    #         tensorboard_embeddings(dataset_visualization[:1500], im_size, model, save_path)  # OOM error when doing entire dataset at once with alexnet
    #         save_path = vis_path + '/best_worst/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         show_n_best_and_worst_configs(dataset_visualization[:1500], im_size, 64, model, save_path=save_path)
    #         save_path = vis_path + '/losses_scores/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         make_losses_and_scores_barplot(dataset_visualization[:1500], model, save_path=save_path)
    #
    #     else:
    #         save_path = vis_path + '/latent_tsne/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         computeTSNEProjectionOfLatentSpace(dataset_visualization, im_size, model, save_path=save_path)
    #         save_path = vis_path + '/tsne_grid/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         twoDimensionalTsneGrid(dataset_visualization[:int(np.floor(np.sqrt(dataset_visualization.shape[0]))**2),:,:,:], im_size, model, 1, int(np.floor(np.sqrt(dataset_visualization.shape[0]))), pixel_or_latent='latent', save_path=save_path)
    #         save_path = vis_path + '/interpolations/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         visualizeInterpolation(dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], model, im_size, save_path=save_path)
    #         save_path = vis_path + '/arithmetics/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         visualizeArithmetics(dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], dataset_visualization[np.random.randint(0, dataset_visualization.shape[0])], model, im_size, save_path=save_path)
    #         save_path = vis_path + '/embeddings/' + model_type + '_' + str(latent_dim) + '_latent_dims'
    #         tensorboard_embeddings(dataset_visualization, im_size, model, save_path)
    #         save_path = vis_path + '/best_worst/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         show_n_best_and_worst_configs(dataset_visualization, im_size, 64, model, save_path=save_path)
    #         save_path = vis_path + '/losses_scores/' + model_type + '_latentdim_' + str(latent_dim) + '_'
    #         make_losses_and_scores_barplot(dataset_visualization, model, save_path=save_path)
    #
    # save_path = vis_path + '/losses_and_scores_mean/' + model_type + '_'
    # if '16_dims' in model_type:
    #     mean_losses_and_scores_over_latent_dims(dataset_visualization[:1500], model_type, chosen_n_units, im_size, save_path=save_path)  # otherwise OOM
    # else:
    #     mean_losses_and_scores_over_latent_dims(dataset_visualization, model_type, chosen_n_units, im_size, save_path=save_path)

