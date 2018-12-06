import itertools
import numpy as np

flat_matrices = np.array(list(itertools.product([0,1], repeat=3*5)))
matrices = np.reshape(flat_matrices, [-1, 3, 5])

for i in range(10000,10003):
    print(matrices[i])