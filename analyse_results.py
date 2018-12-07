import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys, os
from parameters import n_hidden_units_max, im_size, model_type

sys.setrecursionlimit(1500)  # needed to make the recursion limit higher (?)

if model_type is 'conv':
    chosen_n_units = range(8, n_hidden_units_max+1, 4)  # takes too long to compute everything
else:
    chosen_n_units = range(1, n_hidden_units_max+1)

print('loading dataset and simulation results')
dataset = np.load('./dataset.npy')
final_losses_order_all = np.load('./' + model_type +'_final_losses_order_all.npy')

# scores count the avg. position of a configuration: if it has the lowest loss->2**15, if is has the highest loss->1.
# note that the higher the score, the better the performance (this is easier for visualizing the result graphs).
# each line i corresponds to the mean score over all models with n_hidden_units <= i+1 (e.g. line 0 contains the scores
# for the net with a single hidden unit and line 2 contains the avg scores over models with 1, 2 & 3 hidden units.
print('computing mean scores over models with increasing n_hidden_units')
if not os.path.exists('./scores.npy'):
    scores = np.zeros(shape=(final_losses_order_all.shape[0], 2**15))
    for i in range(final_losses_order_all.shape[0]):
        for j in range(2**15):
            scores[i:, j] += np.squeeze(
                np.tile(np.where(j == final_losses_order_all[i, :]), final_losses_order_all.shape[0] - i))
        scores[i, :] /= i + 1
    scores = 2**15+1-scores # originally, the best configs have low values. Switch this for better visualisation.
else:
    scores = np.load('scores.npy')

print('creating graph for final results.')
mean_score = np.mean(final_losses_order_all, axis=0)

ind = np.arange(2**15)
fig, ax = plt.subplots()
ax.bar(ind, mean_score, color=(3./255, 57./255, 108./255))

# add some text for labels, title and axes ticks, and save figure
ax.set_xlabel('configuration IDs')
ax.set_ylabel('Mean scores')
plt.title('Current mean over networks 1 -> ' + str(n_hidden_units_max))
plt.savefig('./' + model_type + '_mean_scores.png')

# plot five best and five worst configs
mean_score_order = mean_score.argsort()
n_samples = 5
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(2, n_samples, index + 1)
    sample_image = dataset[mean_score_order[index], :, :, 0].reshape(im_size[0], im_size[1])
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")
    plt.title('Best configs - rank: (1=BEST): ' + str(index))
    plt.subplot(2, n_samples, n_samples + index + 1)
    sample_image = dataset[mean_score_order[-(index+1)], :, :, 0].reshape(im_size[0], im_size[1])
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")
    plt.title('Worst configs - rank: (1=WORST): ' + str(index))
plt.savefig('./' + model_type + '_mean_scores_best_and_worst_configs.png')

# make a cool gif showing the evolution of mean_score as neurons are added to the hidden layer
print('creating gif of results across networks')
imgs_for_gif = []


def plot_for_offset(data):
    plt.close('all')
    mean_score = np.mean(data, axis=0)
    ind = np.arange(2 ** 15)
    fig, ax = plt.subplots()
    ax.bar(ind, mean_score, color=(3. / 255, 57. / 255, 108. / 255))
    ax.set_xlabel('configuration IDs')
    ax.set_ylabel('Mean scores')
    ax.set_ylim(0, 2**15)
    plt.title('Current mean over networks 1 -> ' + str(data.shape[0]))
    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


# make gif
for i in range(len(chosen_n_units)):
    print("\r{}/{} ({:.1f}%) ".format(i, len(chosen_n_units), i * 100 / len(chosen_n_units)), end="")
    imgs_for_gif.append(plot_for_offset(final_losses_order_all[:i, :]))
imageio.mimsave('./' + model_type + '_mean_scores_evolving.gif', imgs_for_gif, fps=4)