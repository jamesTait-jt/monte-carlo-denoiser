# Total number of images we are training on)
TRAIN_SCENES = 9
TEST_SCENES = 3

# Parameters to go into the patching function
NUM_DARTS = 400
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
PATCH_WIDTH = 60
PATCH_HEIGHT = 60

IMG_SHAPE = (PATCH_WIDTH, PATCH_HEIGHT, 3)
DENOISER_INPUT_SHAPE = (PATCH_WIDTH, PATCH_HEIGHT, 27)

# Do we do the albedo divide step
ALBEDO_DIVIDE = False

# Do we need to generate new patches?
MAKE_NEW_PATCHES = False
LOAD_NEW_IMAGES = False

PATCHES_PATH = "./data/patches/patches_data.pkl"

AUGMENTATION = False

MAKE_NEW_PREDICTION = False
APPLY_NEW_KERNEL = True
