import numpy as np
import itertools
from batchMaker import StimMaker
from parameters import *


final_losses_order_all = np.array([[0,1,2],
                                   [1,0,2]])
n_models = final_losses_order_all.shape[0]
n_matrices = final_losses_order_all.shape[1]
scores = np.zeros(shape=(n_models, n_matrices))
for i in range(n_models):
    for j in range(n_matrices):
        scores[i:, j] += np.squeeze(np.tile(np.where(j == final_losses_order_all[i, :]), final_losses_order_all.shape[0] - i))
    scores[i, :] /= i + 1
scores = n_matrices - scores  # originally, the best configs have low values. Switch this for better visualisation.

print(scores)
print(np.mean(scores, axis=0))
