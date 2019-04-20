import os
from glob import glob
import Imath
import exr
import numpy as np
import pickle
from data import generate_darts
import config
from scipy import ndimage
import matplotlib.pyplot as plt

from keras.preprocessing.image import array_to_img, img_to_array

noisy_test_dir = "../data/tungsten/test/noisy/32"
reference_test_dir = "../data/tungsten/test/reference/1024"
noisy_train_dir = "../data/tungsten/train/noisy/32"
reference_train_dir = "../data/tungsten/train/reference/4096"

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

        # --- ALBEDO DIVIDE --- # 
        albedo = exr.getBuffer(exr_file, "albedo")
        albedo_divided_diffuse = np.divide(diffuse, albedo + 0.00316)

        image_dict[test_or_train][reference_or_noisy]["albedo_divided"].append(albedo_divided_diffuse)

        # --- Reference needs normal for importance sampling --- #
        normal = exr.getBuffer(exr_file, "normal")
        image_dict[test_or_train][reference_or_noisy]["normal"].append(normal)

        if reference_or_noisy == "noisy":

            diffuse_gradx, diffuse_grady = getGrads(diffuse)
            image_dict[test_or_train][reference_or_noisy]["diffuse_gx"].append(diffuse_gradx)
            image_dict[test_or_train][reference_or_noisy]["diffuse_gy"].append(diffuse_grady)

            albedo_divided_gradx, albedo_divided_grady = getGrads(albedo_divided_diffuse)
            image_dict[test_or_train][reference_or_noisy]["albedo_divided_gx"].append(albedo_divided_gradx)
            image_dict[test_or_train][reference_or_noisy]["albedo_divided_gy"].append(albedo_divided_grady)

            diffuse_variance = exr.getBuffer(exr_file, "diffuseVariance")
            diffuse_variance = np.reshape(
                diffuse_variance,
                (diffuse_variance.shape[0], diffuse_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["diffuse_var"].append(diffuse_variance)

            # --- ALBEDO DIVIDE ON VARIANCE --- #
            albedo_divided_variance = np.mean(
                np.divide(diffuse_variance, (albedo + 0.00316) ** 2),
                axis=2
            )
            albedo_divided_variance = np.reshape(
                albedo_divided_variance,
                (albedo_divided_variance.shape[0], albedo_divided_variance.shape[1], 1)
            )
            image_dict[test_or_train][reference_or_noisy]["albedo_divided_var"].append(albedo_divided_variance)

            # --- NORMAL --- #
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
                "albedo_divided" : [],
                "normal" : []
            },
            "noisy" : {
                "diffuse" : [],
                "diffuse_gx" : [],
                "diffuse_gy" : [],
                "diffuse_var" : [],
                "albedo_divided" : [],
                "albedo_divided_gx" : [],
                "albedo_divided_gy" : [],
                "albedo_divided_var" : [],
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
                "albedo_divided" : [],
                "normal" : []
            },
            "noisy" : {
                "diffuse" : [],
                "diffuse_gx" : [],
                "diffuse_gy" : [],
                "diffuse_var" : [],
                "albedo_divided" : [],
                "albedo_divided_gx" : [],
                "albedo_divided_gy" : [],
                "albedo_divided_var" : [],
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
        print("\nDumping data...")
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

def importanceSample(full_dict, patches):
    # build the metric map
    metrics = ['relvar', 'variance']
    weights = [1.0, 1.0]

    diffuse = full_dict["train"]["reference"]["diffuse"]
    normal = full_dict["train"]["reference"]["normal"]
    print("\nSampling training patches...")
    for scene_num in range(len(diffuse)):
        fillSampledPatches(
            full_dict,
            diffuse,
            normal,
            "train",
            scene_num,
            metrics,
            weights,
            patches
        )

    diffuse = full_dict["test"]["noisy"]["diffuse"]
    normal = full_dict["test"]["noisy"]["normal"]
    print("Sampling test patches...")
    for scene_num in range(len(diffuse)):
        fillSampledPatches(
            full_dict,
            diffuse,
            normal,
            "test",
            scene_num,
            metrics,
            weights,
            patches
        )


def fillSampledPatches(
    full_dict,
    diffuse,
    normal,
    test_or_train,
    scene_num,
    metrics,
    weights,
    patches
):
    diffuse_buf = diffuse[scene_num]
    normal_buf = normal[scene_num]
    importance_map = getImportanceMap([diffuse_buf, normal_buf], metrics, weights)

    # Save importance map
    if test_or_train == "test":
        plt.figure()
        imgplot = plt.imshow(importance_map)
        imgplot.axes.get_xaxis().set_visible(False)
        imgplot.axes.get_yaxis().set_visible(False)
        plt.savefig("../data/output/{0}/sampling/importance_map.png".format(scene_num))

    patch_positions = samplePatches(diffuse_buf.shape[:2])
    
    # Save original patch positions
    if test_or_train == "test":
        plt.figure()
        plt.scatter(list(a[0] for a in patch_positions), list(a[1] for a in patch_positions))
        plt.savefig("../data/output/{0}/sampling/dart_thrown_patches.png".format(scene_num))

    selection = diffuse_buf * 0.1
    for i in range(patch_positions.shape[0]):
        x, y = patch_positions[i, 0], patch_positions[i, 1]
        selection[y : y + config.PATCH_WIDTH, x : x + config.PATCH_HEIGHT, :] = \
            diffuse_buf[y : y + config.PATCH_WIDTH, x : x + config.PATCH_HEIGHT, :]
            
    pad = config.PATCH_WIDTH // 2
    pruned = np.maximum(
        0,
        prunePatches(
            diffuse_buf.shape[:2],
            patch_positions + pad,
            importance_map
        ) - pad
    )
            
    selection = diffuse_buf * 0.1
    for i in range(pruned.shape[0]):
        x, y = pruned[i, 0], pruned[i, 1]
        selection[y : y + config.PATCH_WIDTH, x : x + config.PATCH_HEIGHT, :] = \
            diffuse_buf[y : y + config.PATCH_WIDTH, x : x + config.PATCH_HEIGHT, :]

    # Save pruned patch positions
    if test_or_train == "test":
        plt.figure()
        plt.scatter(list(a[0] for a in pruned), list(a[1] for a in pruned))
        plt.savefig("../data/output/{0}/sampling/pruned_patches.png".format(scene_num))


    patch_positions = pruned + pad

    cropPatches(full_dict, patch_positions, patches, scene_num, test_or_train)


def getImportanceMap(img_buffers, metrics, weights):
    importance_map = None
    for buf, metric, weight in zip(img_buffers, metrics, weights):
        if metric == 'variance':
            cur = getVarianceMap(buf, relative=False)
        elif metric == 'relvar':
            cur = getVarianceMap(buf, relative=True)
        else:
            print('Unexpected metric:', metric)
        if importance_map is None:
            importance_map = cur*weight
        else:
            importance_map += cur*weight
    return importance_map / importance_map.max()


def getVarianceMap(img_buffer, relative=False):

     # introduce a dummy third dimension if needed
    if img_buffer.ndim < 3:
        img_buffer = img_buffer[:, :, np.newaxis]

    # compute variance
    mean = ndimage.uniform_filter(
        img_buffer,
        size=(config.PATCH_WIDTH, config.PATCH_HEIGHT, 1)
    )

    sqrmean = ndimage.uniform_filter(
        img_buffer ** 2,
        size=(config.PATCH_WIDTH, config.PATCH_HEIGHT, 1)
    )

    variance = np.maximum(sqrmean - mean ** 2, 0)

    # convert to relative variance if requested
    if relative:
        variance = variance/np.maximum(mean ** 2, 1e-2)

    # take the max variance along the three channels, gamma correct it to get a
    # less peaky map, and normalize it to the range [0,1]
    variance = variance.max(axis=2)
    variance = np.minimum(variance ** (1.0 / 2.2), 1.0)

    return variance/variance.max()

# Sample patches using dart throwing (works well for sparse/non-overlapping patches)
def samplePatches(img_dim, maxiter=5000):

    # estimate each sample patch area
    full_area = float(img_dim[0] * img_dim[1])
    sample_area = full_area/config.NUM_DARTS

    # get corresponding dart throwing radius
    radius = np.sqrt(sample_area / np.pi)
    minsqrdist = (2 * radius) ** 2

    # compute the distance to the closest patch
    def get_sqrdist(x, y, patches):
        if len(patches) == 0:
            return np.infty
        dist = patches - [x, y]
        return np.sum(dist ** 2, axis=1).min()

    # perform dart throwing, progressively reducing the radius
    rate = 0.96
    patches = np.zeros((config.NUM_DARTS,2), dtype=int)
    xmin, xmax = 0, img_dim[1] - config.PATCH_HEIGHT - 1
    ymin, ymax = 0, img_dim[0] - config.PATCH_WIDTH - 1
    for patch in range(config.NUM_DARTS):
        done = False
        while not done:
            for i in range(maxiter):
                x = np.random.randint(xmin, xmax)
                y = np.random.randint(ymin, ymax)
                sqrdist = get_sqrdist(x, y, patches[:patch, :])
                if sqrdist > minsqrdist:
                    patches[patch, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2 * radius) ** 2

    return patches


def prunePatches(shape, patches, imp):

    pruned = np.empty_like(patches)

    # Generate a set of regions tiling the image using snake ordering.
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            if y//step % 2 == 0:
                xrange = range(0, shape[1], step)
            else:
                xrange = reversed(range(0, shape[1], step))
            for x in xrange:
                regions.append((x, x + step, y, y + step))
        return regions

    # Split 'patches' in current and remaining sets, where 'cur' holds the
    # patches in the requested region, and 'rem' holds the remaining patches.
    def split_patches(patches, region):
        cur = np.empty_like(patches)
        rem = np.empty_like(patches)
        ccount, rcount = 0, 0
        for i in range(patches.shape[0]):
            x, y = patches[i, 0], patches[i, 1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur[ccount, :] = [x, y]
                ccount += 1
            else:
                rem[rcount, :] = [x, y]
                rcount += 1
        return cur[ : ccount, :], rem[ : rcount, :]

    # Process all patches, region by region, pruning them randomly according to
    # their importance value, ie. patches with low importance have a higher
    # chance of getting pruned. To offset the impact of the binary pruning
    # decision, we propagate the discretization error and take it into account
    # when pruning.
    rem = np.copy(patches)
    count, error = 0, 0
    for region in get_regions_list(shape, 4 * config.PATCH_WIDTH):
        cur, rem = split_patches(rem, region)
        for i in range(cur.shape[0]):
            x, y = cur[i, 0], cur[i, 1]
            if imp[y, x] - error > np.random.random():
                pruned[count, :] = [x, y]
                count += 1
                error += 1 - imp[y, x]
            else:
                error += 0 - imp[y, x]

    return pruned[: count, :]


# crops all channels
def cropPatches(data, darts, patches, img_number, test_or_train):
    half_patch = config.PATCH_WIDTH // 2
    sx, sy = half_patch, half_patch
    for test_or_train_key in data:
        if test_or_train_key == test_or_train:
            for noisy_or_reference in data[test_or_train]:
                for key, val in data[test_or_train][noisy_or_reference].items():
                    for dart in darts:
                        dart = tuple(dart)
                        px, py = dart
                        patch = val[img_number][(py - sy) : (py + sy), (px - sx) : (px + sx), :]
                        patches[test_or_train][noisy_or_reference][key].append(patch)
            
def getSampledPatches():
    if config.LOAD_NEW_IMAGES:
        print("\nReading exr files...")
        full_dict = getAllImagesAndSaveAsPkl()
    else:
        print("\nLoading buffers from disk...")
        full_dict = loadPkl(full_pkl_path)
        #for test_or_train in full_dict:
        #    for noisy_or_reference in full_dict[test_or_train]:
        #        for key, val in full_dict[test_or_train][noisy_or_reference].items():
        #            print(key)
        #            print(np.amax(val[0]))
        #            print(np.amin(val[0]))
    
    if config.MAKE_NEW_PATCHES:
        patches = initialiseDict()
        importanceSample(full_dict, patches)
        del full_dict
        print("\nTraining patches: %d" % len(patches["train"]["noisy"]["diffuse"]))
        print("Test patches: %d" % len(patches["test"]["reference"]["diffuse"]))
        saveAsPkl(patches, patches_pkl_path)
    else:
        del full_dict
        patches = loadPkl(patches_pkl_path)

    return patches
