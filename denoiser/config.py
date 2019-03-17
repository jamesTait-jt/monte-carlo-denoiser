# Total number of images we are training on)
TOTAL_SCENES = 3

# Parameters to go into the patching function
NUM_DARTS = 400
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
PATCH_WIDTH = 64 
PATCH_HEIGHT = 64

# We differentiate all of the buffers to highlight features for the network.
# This list contains the paths of the images that will be differentiated
IMAGE_PATHS_TO_DIFFERENTIATE = [
    "data/full/reference_colour.png",
    "data/full/noisy_colour.png",

    "data/full/reference_sn.png",
    "data/full/noisy_sn.png",

    "data/full/reference_albedo.png",
    "data/full/noisy_albedo.png",

    "data/full/reference_depth.png",
    "data/full/noisy_depth.png"
]

# Save locations of the differentiated images
DIFFERENTIATED_SAVE_DIRS = [
    "data/full/reference_colour_gradx.png",
    "data/full/reference_colour_grady.png",

    "data/full/noisy_colour_gradx.png",
    "data/full/noisy_colour_grady.png",

    "data/full/reference_sn_gradx.png",
    "data/full/reference_sn_grady.png",

    "data/full/noisy_sn_gradx.png",
    "data/full/noisy_sn_grady.png",

    "data/full/reference_albedo_gradx.png",
    "data/full/reference_albedo_grady.png",

    "data/full/noisy_albedo_gradx.png",
    "data/full/noisy_albedo_grady.png",

    "data/full/reference_depth_gradx.png",
    "data/full/reference_depth_grady.png",

    "data/full/noisy_depth_gradx.png",
    "data/full/noisy_depth_grady.png",
]

# The image paths containing the full buffers. Colour, features and gradients as
# well as their corresponding variances
FULL_IMAGE_PATHS = [
    "data/full/reference_colour.png",
    "data/full/noisy_colour.png",

    "data/full/reference_colour_gradx.png",
    "data/full/noisy_colour_gradx.png",

    "data/full/reference_colour_grady.png",
    "data/full/noisy_colour_grady.png",

    "data/full/reference_sn.png",
    "data/full/noisy_sn.png",

    "data/full/reference_sn_gradx.png",
    "data/full/noisy_sn_gradx.png",

    "data/full/reference_sn_grady.png",
    "data/full/noisy_sn_grady.png",

    "data/full/reference_albedo.png",
    "data/full/noisy_albedo.png",

    "data/full/reference_albedo_gradx.png",
    "data/full/noisy_albedo_gradx.png",

    "data/full/reference_albedo_grady.png",
    "data/full/noisy_albedo_grady.png",

    "data/full/reference_depth.png",
    "data/full/noisy_depth.png",

    "data/full/reference_depth_gradx.png",
    "data/full/noisy_depth_gradx.png",

    "data/full/reference_depth_grady.png",
    "data/full/noisy_depth_grady.png",

    "data/full/reference_colour_vars.png",
    "data/full/noisy_colour_vars.png",

    "data/full/reference_sn_vars.png",
    "data/full/noisy_sn_vars.png",

    "data/full/reference_albedo_vars.png",
    "data/full/noisy_albedo_vars.png",

    "data/full/reference_depth_vars.png",
    "data/full/noisy_depth_vars.png"
]

# Directories where the patches will be saved
PATCH_SAVE_DIRS = [
    "data/patches/reference_colour/",
    "data/patches/noisy_colour/",
    "data/patches/noisy_colour_gradx/",
    "data/patches/noisy_colour_grady/",
    "data/patches/noisy_colour_vars/",
    "data/patches/noisy_sn_gradx/",
    "data/patches/noisy_sn_grady/",
    "data/patches/noisy_sn_vars/",
    "data/patches/noisy_albedo_gradx/",
    "data/patches/noisy_albedo_grady/",
    "data/patches/noisy_albedo_vars/",
    "data/patches/noisy_depth_gradx/",
    "data/patches/noisy_depth_grady/",
    "data/patches/noisy_depth_vars/"
]
