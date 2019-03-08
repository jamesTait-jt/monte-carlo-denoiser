import numpy as np
import config
from scipy import ndimage

from keras.preprocessing.image import array_to_img, img_to_array, load_img

def toColourVal(x):
    x = float(x)
    if x < 0:
        x = 0
    elif x > 1:
        x = 1
    return x

def preProcessReferenceColour():
    with open("data/full/reference_colour.txt") as f:
        data = np.array([[toColourVal(x.split(' ')[0]), toColourVal(x.split(' ')[1]), toColourVal(x.split(' ')[2])] for x in f.read().split(',')[:-1]])
        data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    return data

def preProcessNoisyColour():
    with open("data/full/noisy_colour.txt") as f:
        colour_data = np.array([[toColourVal(x.split(' ')[0]), toColourVal(x.split(' ')[1]), toColourVal(x.split(' ')[2])] for x in f.read().split(',')[:-1]])
        colour_data = np.reshape(colour_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))
    
        img = array_to_img(colour_data)        
        gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
        grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0

    with open("data/full/noisy_colour_vars.txt") as f:
        var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
        var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
        var_data = var_data / np.amax(var_data)

    return colour_data, gradx, grady, var_data

def preProcessAlbedo():
    with open("data/full/noisy_albedo.txt") as f:
        data = np.array([[float(x.split(' ')[0]), float(x.split(' ')[1]), float(x.split(' ')[2])] for x in (f.read().split(',')[:-1])])
        data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))

        img = array_to_img(data)        
        gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
        grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0

    with open("data/full/noisy_albedo_vars.txt") as f:
        var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
        var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
        var_data = var_data / np.amax(var_data)

    return gradx, grady, var_data

# The depths are normalised between 0 and 1
def preProcessDepth():
    with open("data/full/noisy_depth.txt") as f:
        data = np.array([float(x) for x in f.read().split(',')[:-1]])
        data /= np.max(data)
        data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))

        img = array_to_img(data)        
        gradx = ndimage.sobel(img, axis=0, mode='constant')
        grady = ndimage.sobel(img, axis=1, mode='constant')

        gradx = np.reshape(gradx, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0
        grady = np.reshape(grady, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)) / 255.0

    with open("data/full/noisy_depth_vars.txt") as f:
        var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
        var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))
        var_data = var_data / np.amax(var_data)

    return gradx, grady, var_data

# Nothing special is done to the surface normals
def preProcessSurfaceNormal():
    with open("data/full/noisy_sn.txt") as f:
        data = np.array([[float(x.split(' ')[0]), float(x.split(' ')[1]), float(x.split(' ')[2])] for x in (f.read().split(',')[:-1])])
        data = np.reshape(data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3))

        img = array_to_img(data)        
        gradx = ndimage.sobel(img, axis=0, mode='constant') / 255.0
        grady = ndimage.sobel(img, axis=1, mode='constant') / 255.0

    with open("data/full/noisy_sn_vars.txt") as f:
        var_data = np.array([float(x) for x in f.read().split(',')[:-1]])
        var_data = np.reshape(var_data, (config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1))

        var_data = var_data / np.amax(var_data)

    return gradx, grady, var_data

noisy_colour, noisy_colour_gradx, noisy_colour_grady, noisy_colour_var = preProcessNoisyColour()
sn_gradx, sn_grady, sn_var = preProcessSurfaceNormal()
albedo_gradx, albedo_grady, albedo_var = preProcessAlbedo()
depth_gradx, depth_grady, depth_var = preProcessDepth()

full_images = {
    "reference_colour" : preProcessReferenceColour(),
    "noisy_colour" : noisy_colour,
    "noisy_colour_gradx" : noisy_colour_gradx,
    "noisy_colour_grady" : noisy_colour_grady,
    "noisy_colour_var" : noisy_colour_var,
    "noisy_sn_gradx" : sn_gradx,
    "noisy_sn_grady" : sn_gradx,
    "noisy_sn_var" : sn_var,
    "noisy_albedo_gradx" : albedo_gradx,
    "noisy_albedo_grady" : albedo_gradx,
    "noisy_albedo_var" : albedo_var,
    "noisy_depth_gradx" : depth_gradx,
    "noisy_depth_grady" : depth_gradx,
    "noisy_depth_var" : depth_var
}

for key in full_images:
    array_to_img(full_images[key]).save("data/full/" + key + ".png")
