# Total number of images we are training on)
TRAIN_SCENES = 8
TEST_SCENES = 4

# Parameters to go into the patching function
NUM_DARTS = 400
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
PATCH_WIDTH = 60
PATCH_HEIGHT = 60

# Do we do the albedo divide step
ALBEDO_DIVIDE = False

# Do we need to generate new patches?
MAKE_NEW_PATCHES = False
PATCHES_PATH = "./data/patches/patches_data.pkl"

LOAD_NEW_IMAGES = False

AUGMENTATION = False
