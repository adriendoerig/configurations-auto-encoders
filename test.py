import numpy as np

final_losses_order_all = np.array([[0, 2, 1],
                                   [1, 2, 0],
                                   [0, 1, 2]])

scores = np.zeros(shape=(final_losses_order_all.shape[0], 3))
for i in range(final_losses_order_all.shape[0]):
    for j in range(3):
        scores[i:, j] += np.squeeze(np.tile(np.where(j==final_losses_order_all[i, :]), final_losses_order_all.shape[0]-i))
    scores[i,  :] /= i+1

print(final_losses_order_all)
print(scores)
scores = 3 - 1 - scores  # originally, the best configs have low values. Switch this for better visualisation.
print(scores)