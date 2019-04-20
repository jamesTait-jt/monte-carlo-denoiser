import sys
import keras
import numpy as np
import math
import tensorflow as tf
from keras.preprocessing.image import array_to_img

import config
import eval_util
import tungsten_data

# Parse arguments
model_dir = sys.argv[1]
test_img_index = int(sys.argv[2])
model_type = sys.argv[3]

test_img_name = eval_util.scenes_dict[test_img_index]
save_dir = "../experiments/test_scenes/{}/".format(test_img_name)

full_pkl_path = "../data/tungsten/pkl/full_dict.pkl"
images = tungsten_data.loadPkl(full_pkl_path)

# Save the image buffers as images
print("Saving buffers as images...")
eval_util.saveBuffers(images, save_dir, test_img_index)
print("Done!\n")

feature_list = ["normal", "albedo", "depth"]
buffers_in, noisy_img = eval_util.getModelInput(images, test_img_index, feature_list)

print("Loading model...")
old_model = keras.models.load_model(sys.argv[1], compile=False)
print("Done!\n")

# Pop the input layer off so we can make a new input layer that is the shape of
# the full image (not patches)
old_model.layers.pop(0)

# Defining new model shape
new_input = keras.layers.Input(buffers_in.shape)
new_output = old_model(new_input)
new_model = keras.models.Model(new_input, new_output)

print("Making prediction...")
buffers_in = np.expand_dims(buffers_in, 0)
noisy_img = np.expand_dims(noisy_img, 0)
weights = new_model.predict(buffers_in)
print("Done!\n")

print("Applying weights...")
pred = eval_util.applyKernel(noisy_img, weights)
print("Done!\n")

if config.ALBEDO_DIVIDE:
    pred = eval_util.albedoMultiply(images, pred, test_img_index) 

pred = pred.clip(0, 1)[0]
pred = array_to_img(pred)
pred.save(save_dir + "predictions/" + model_type + ".png")
