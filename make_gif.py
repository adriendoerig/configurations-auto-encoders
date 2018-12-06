import numpy as np
import imageio
import matplotlib.pyplot as plt
import sys
from parameters import n_hidden_units_max

sys.setrecursionlimit(1500)  # need to make the recursion limit higher

dataset = np.load('./dataset.npy')
final_losses_order_all = np.load('./final_losses_order_all.npy')
final_losses_order_all = 2**15-final_losses_order_all # originally, the best configs have low values. Switch this for better visualisation.

# make a cool gif showing the evolution of mean_score as neurons are added to the hidden layer
imgs_for_gif = []
def plot_for_offset(data):
    mean_score = np.mean(data, axis=0)
    ind = np.arange(2 ** 15)
    fig, ax = plt.subplots()
    ax.bar(ind, mean_score, color=(3. / 255, 57. / 255, 108. / 255))
    ax.set_xlabel('configuration IDs')
    ax.set_ylabel('Mean scores')
    ax.set_ylim(0, 2**15)
    # Used to return the plot as an image array
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image

# make gif
plot_for_offset(final_losses_order_all[:2, :])
imageio.mimsave('./mean_scores_evolving.gif', [plot_for_offset(final_losses_order_all[:img+1, :]) for img in range(2)], fps=4)