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

print("Loading model...")
old_model = keras.models.load_model(sys.argv[1], compile=False)
print("Done!\n")

pred = eval_util.denoiseFullTestImg(old_model, images, test_img_index)
pred.save(save_dir + "predictions/" + model_type + ".png")
