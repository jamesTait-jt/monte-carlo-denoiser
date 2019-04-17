import tungsten_data
from keras.preprocessing.image import array_to_img

print("Loading in image buffers...")
all_imgs = tungsten_data.loadPkl(tungsten_data.full_pkl_path)

print("Saving image buffers...")
for test_or_train in all_imgs:
    if test_or_train == "test":
        for noisy_or_ref in all_imgs[test_or_train]:
            if noisy_or_ref == "noisy":
                for key, img_buffers in all_imgs[test_or_train][noisy_or_ref].items():
                    for index in range(len(img_buffers)):
                        img_buffer = img_buffers[index]
                        array_to_img(img_buffer).save("../data/output/{0}/features/".format(index) + key + ".png")

print("Importance sampling...")
patches = tungsten_data.initialiseDict()
tungsten_data.importanceSample(all_imgs, patches)
