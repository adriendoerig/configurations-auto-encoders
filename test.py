import numpy as np
import itertools
from batchMaker import StimMaker
from parameters import *

n_squares = 9
flat_matrices = np.array(list(itertools.product([0, 1], repeat=3 * 5)))
matrices = np.reshape(flat_matrices, [-1, 3, 5])
flat_matrices[:, 7] = 0
# remove entries with the wrong number of squares
row_sums = np.sum(flat_matrices, axis=1)
flat_matrices = flat_matrices[row_sums==15-n_squares, :]
unique_flat_matrices = np.unique(flat_matrices, axis=0)
matrices = np.reshape(unique_flat_matrices, [-1, 3, 5])
matrices[:, 1, 2] = 0



print(matrices)