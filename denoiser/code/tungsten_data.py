import os
from glob import glob
import Imath
import exr
import numpy as np
import pickle
from data import generate_darts
import config

from keras.preprocessing.image import array_to_img, img_to_array

noisy_test_dir = "../data/tungsten/test/noisy"
reference_test_dir = "../data/tungsten/test/reference"
noisy_train_dir = "../data/tungsten/train/noisy"
reference_train_dir = "../data/tungsten/train/reference"

full_pkl_path = "../data/tungsten/pkl/full_dict.pkl"
patches_pkl_path = "../data/tungsten/pkl/patches_dict.pkl"

def getScenes(input_dir, image_dict, reference_or_noisy, test_or_train):
    input_files = glob(os.path.join(input_dir, "*.exr"))
    input_files.sort()
    file_index = 1

    for curr_file in input_files:
        print("\nWorking on image %d of %d" % (file_index, len(input_files)))

        # Open the exr file and get its header
        exr_file = exr.open(curr_file)

        # --- DIFFUSE --- #
        diffuse = exr.getBuffer(exr_file, "diffuse")
        diffuse = diffuse.clip(0, 1) # Clip the diffuse colour values between 0 and 1
        image_dict[test_or_train][reference_or_noisy]["diffuse"].append(diffuse)

        if reference_or_noisy == "noisy":

            diffuse_gradx, diffuse_grady = getGrads(diffuse)
            image_dict[test_or_train][reference_or_noisy]["diffuse_gx"].append(diffuse_gradx)
            image_dict[test_or_train][reference_or_noisy]["diffuse_gy"].append(diffuse_grady)

            diffuse_variance = exr.getBuffer(exr_file, "diffuseVariance")
            diffuse_variance = np.reshape(
                diffuse_variance,
                (diffuse_variance.shape[0], diffuse_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["diffuse_var"].append(diffuse_variance)

            # --- NORMAL --- #
            normal = exr.getBuffer(exr_file, "normal")
            image_dict[test_or_train][reference_or_noisy]["normal"].append(normal)

            normal_gradx, normal_grady = getGrads(normal)
            image_dict[test_or_train][reference_or_noisy]["normal_gx"].append(normal_gradx)
            image_dict[test_or_train][reference_or_noisy]["normal_gy"].append(normal_grady)

            normal_variance = exr.getBuffer(exr_file, "normalVariance")
            normal_variance = np.reshape(
                normal_variance,
                (normal_variance.shape[0], normal_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["normal_var"].append(normal_variance)

            # --- ALBEDO --- #
            albedo = exr.getBuffer(exr_file, "albedo")
            image_dict[test_or_train][reference_or_noisy]["albedo"].append(albedo)

            albedo_gradx, albedo_grady = getGrads(albedo)
            image_dict[test_or_train][reference_or_noisy]["albedo_gx"].append(albedo_gradx)
            image_dict[test_or_train][reference_or_noisy]["albedo_gy"].append(albedo_grady)

            albedo_variance = exr.getBuffer(exr_file, "albedoVariance")
            albedo_variance = np.reshape(
                albedo_variance,
                (albedo_variance.shape[0], albedo_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["albedo_var"].append(albedo_variance)

            # --- DEPTH --- #
            depth = exr.getBuffer(exr_file, "depth")
            depth = np.clip(depth, 0, np.amax(depth)) # Normalise depth values between 0 and 1
            depth /= np.amax(depth)
            depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
            image_dict[test_or_train][reference_or_noisy]["depth"].append(depth)

            depth_gradx, depth_grady = getGrads(depth)
            image_dict[test_or_train][reference_or_noisy]["depth_gx"].append(depth_gradx)
            image_dict[test_or_train][reference_or_noisy]["depth_gy"].append(depth_grady)

            depth_variance = exr.getBuffer(exr_file, "depthVariance")
            depth_variance = np.reshape(
                depth_variance,
                (depth_variance.shape[0], depth_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["depth_var"].append(depth_variance)

        file_index += 1

def initialiseDict():
    image_dict = {
        "train" : {
            "reference" : {
                "diffuse" : [],
            },
            "noisy" : {
                "diffuse" : [],
                "diffuse_gx" : [],
                "diffuse_gy" : [],
                "diffuse_var" : [],
                "normal" : [],
                "normal_gx" : [],
                "normal_gy" : [],
                "normal_var" : [],
                "albedo" : [],
                "albedo_gx" : [],
                "albedo_gy" : [],
                "albedo_var" : [],
                "depth" : [],
                "depth_gx" : [],
                "depth_gy" : [],
                "depth_var" : []
            }
        },
        "test" : {
            "reference" : {
                "diffuse" : [],
            },
            "noisy" : {
                "diffuse" : [],
                "diffuse_gx" : [],
                "diffuse_gy" : [],
                "diffuse_var" : [],
                "normal" : [],
                "normal_gx" : [],
                "normal_gy" : [],
                "normal_var" : [],
                "albedo" : [],
                "albedo_gx" : [],
                "albedo_gy" : [],
                "albedo_var" : [],
                "depth" : [],
                "depth_gx" : [],
                "depth_gy" : [],
                "depth_var" : []
            }
        }
    }
    return image_dict

def getGrads(feature_buffer):
    gradx = np.gradient(feature_buffer, axis=0)
    grady = np.gradient(feature_buffer, axis=1)
    return gradx, grady

def getAllImagesAndSaveAsPkl():
    image_dict = initialiseDict()
    print("\n --- Getting noisy test data --- ")
    getScenes(noisy_test_dir, image_dict, "noisy", "test")
    print("\n --- Getting noisy train data --- ")
    getScenes(noisy_train_dir, image_dict, "noisy", "train")
    print("\n --- Getting reference test data --- ")
    getScenes(reference_test_dir, image_dict, "reference", "test")
    print("\n --- Getting reference train data --- ")
    getScenes(reference_train_dir, image_dict, "reference", "train")
    saveAsPkl(image_dict, full_pkl_path)
    return image_dict

def saveAsPkl(data, filepath):
    with open(filepath, "wb") as f:
        print("Dumping data...")
        pickle.dump(data, f)
        filesize = os.path.getsize(filepath)
        filesize = int(filesize) / 1000000
        print("Done! - data size is: " + str(filesize) + "MB")

def loadPkl(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        filesize = os.path.getsize(filepath)
        filesize = int(filesize) / 1000000
        print("Done! - data size is: " + str(filesize) + "MB")
    return data

def getImageShapes(img_dict):
    test_shapes = []
    diffuse_images = img_dict["test"]["noisy"]["diffuse"]
    for i in range(len(diffuse_images)):
        test_shapes.append(diffuse_images[i].shape)

    train_shapes = []
    diffuse_images = img_dict["train"]["noisy"]["diffuse"]
    for i in range(len(diffuse_images)):
        train_shapes.append(diffuse_images[i].shape)

    return test_shapes, train_shapes

def getDartsArrs(num_darts, full_dict, patch_size):
    test_shapes, train_shapes = getImageShapes(full_dict)
    test_darts = []
    for i in range(len(test_shapes)):
        test_darts.append(generate_darts(
            num_darts,
            test_shapes[i][1],
            test_shapes[i][0],
            patch_size,
            patch_size
        ))
        
    train_darts = []
    for i in range(len(train_shapes)):
        train_darts.append(generate_darts(
            num_darts,
            train_shapes[i][1],
            train_shapes[i][0],
            patch_size,
            patch_size
        ))

    return train_darts, test_darts

def makePatches(full_dict):
    print("Generating patches...")
    train_darts, test_darts = getDartsArrs(config.NUM_DARTS, full_dict, config.PATCH_HEIGHT)
    patches = initialiseDict()
    for test_or_train in full_dict:
        darts = train_darts
        if test_or_train == "test":
            darts = test_darts
        for noisy_or_reference in full_dict[test_or_train]:
            for feature_buffer_k, feature_buffer_v in full_dict[test_or_train][noisy_or_reference].items():
                for i in range(len(feature_buffer_v)):
                    img_array = feature_buffer_v[i]
                    for dart in darts[i]:
                        patch = np.zeros((config.PATCH_HEIGHT, config.PATCH_WIDTH, img_array.shape[2]))
                        for x in range(0, config.PATCH_HEIGHT):
                            for y in range(0, config.PATCH_WIDTH):
                                patch[x][y] = np.array(img_array[dart[0] + x][dart[1] + y])
                        patches[test_or_train][noisy_or_reference][feature_buffer_k].append(patch)
    print("Done!")
    return patches

def getPatches():
    if config.LOAD_NEW_IMAGES:
        full_dict = getAllImagesAndSaveAsPkl()
        patches = makePatches(full_dict)
        del full_dict
        saveAsPkl(patches, patches_pkl_path)
    elif config.MAKE_NEW_PATCHES:
        full_dict = loadPkl(full_pkl_path)
        patches = makePatches(full_dict)
        del full_dict
        saveAsPkl(patches, patches_pkl_path)
    else:
        patches = loadPkl(patches_pkl_path)
    return patches
