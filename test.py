import numpy as np
import itertools
from batchMaker import StimMaker
from parameters import *

stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
flat_matrices[:, 7] = 0
unique_flat_matrices = np.unique(flat_matrices, axis=0)
matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
matrices[:, 1, 2] = 0

for i in range(20):
    stim_maker.showBatch(1, matrices[np.random.randint(16000), :, :] * (other_shape_ID - 1) + 1)