import numpy as np
import math

def py_weighted_average(weights, noisy_img):
    batch_size = weights.shape[0]
    weights_width = weights.shape[1]
    weights_height = weights.shape[2]
    kernel_size = int(math.sqrt(weights.shape[3]))

    prediction_img = np.zeros((batch_size, weights_width, weights_height, 3))
    for batch in range(batch_size):
        print(str(batch) + "/" + str(batch_size))
        for i in range(weights_width):
            for j in range(weights_height):
                for k1 in range(kernel_size):
                    for k2 in range(kernel_size):
                        index_in_patch = kernel_size * k1 + k2; 
                        weight = weights[batch, i, j, index_in_patch];
                        prediction_img[batch, i, j, 0] += weight * noisy_img[batch, i + k1, j + k2, 0];
                        prediction_img[batch, i, j, 1] += weight * noisy_img[batch, i + k1, j + k2, 1];
                        prediction_img[batch, i, j, 2] += weight * noisy_img[batch, i + k1, j + k2, 2];
    return prediction_img
