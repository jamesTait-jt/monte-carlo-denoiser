from keras.applications import VGG19
import keras

import config

def buildVGG():
    """Build and return the vgg network for feature extraction."""
    vgg19 = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=config.IMG_SHAPE
    )
    vgg19.trainable = False
    for layer in vgg19.layers:
        layer.trainable = False

    feature_extractor = keras.models.Model(
        inputs=vgg19.input,
        outputs=vgg19.get_layer("block2_conv2").output
    )

    feature_extractor.trainable = False
    return feature_extractor


def buildGenerator(kernel_size, layers, bn, kpcn, kpcn_size):
    """Build and return the generator for use in the gan."""
    def convLayer(c_input, num_filters):
        """Helper function for a convolutional block."""
        c_output = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            use_bias=False,
            strides=[1, 1],
            padding="SAME",
            kernel_initializer=keras.initializers.glorot_normal(seed=5678)
        )(c_input)
        return c_output

    ################################################

    # The generator takes a noisy image as input
    noisy_img = keras.layers.Input(config.DENOISER_INPUT_SHAPE, name="Generator_input")

    # 9 fully convolutional layers
    x = convLayer(noisy_img, 100)
    x = keras.layers.ReLU()(x)
    for _ in range(layers):
        x = convLayer(x, 100)
        if bn:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    # Final layer is not activated
    if kpcn:
        weights = convLayer(x, pow(kpcn_size, 2))
    else:
        weights = convLayer(x, 3)

    return keras.models.Model(noisy_img, weights, name="Generator")

def buildCritic():
    """Build and return the critic for use in th GAN."""
    def convBlock(c_input, num_filters, strides):
        """Helper function for a convolutional block. Includes LeakyReLU
        activation."""
        c_output = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            strides=strides,
            padding="SAME"
        )(c_input)
        
        output = keras.layers.LeakyReLU(alpha=0.2)(c_output)
        
        return output

    ################################################

    img = keras.layers.Input(shape=config.IMG_SHAPE, name="Critic_input")
    
    x = convBlock(img, 64, strides=[1, 1])
    x = convBlock(img, 64, strides=[2, 2])
    x = keras.layers.Dropout(0.4)(x)
    x = convBlock(x, 128, strides=[1, 1])
    x = convBlock(x, 128, strides=[2, 2])
    x = keras.layers.Dropout(0.4)(x)
    x = convBlock(x, 256, strides=[1, 1])
    x = convBlock(x, 256, strides=[2, 2])
    x = keras.layers.Dropout(0.4)(x)
    x = convBlock(x, 512, strides=[1, 1])
    x = convBlock(x, 512, strides=[2, 2])
    x = keras.layers.Dropout(0.4)(x)

    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1)(x)

    return keras.models.Model(img, x, name="Critic")


def buildDiscriminator():
    """Build and return the discriminator for use in vanilla GAN."""
    def convBlock(c_input, num_filters, strides, bn=True):
        """Helper function for a convolutional block."""
        c_output = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=[3, 3],
            strides=strides,
            padding="SAME"
        )(c_input)

        output = keras.layers.LeakyReLU(alpha=0.2)(c_output)

        if bn:
            output = keras.layers.BatchNormalization(momentum=0.8)(output)

        return output

    ################################################

    img = keras.layers.Input(shape=self.img_shape, name="Discriminator_input")

    x = convBlock(img, 64, strides=[1, 1], bn=False)
    x = convBlock(x, 64, strides=[2, 2])
    x = convBlock(x, 128, strides=[1, 1])
    x = convBlock(x, 128, strides=[2, 2])
    x = convBlock(x, 256, strides=[1, 1])
    x = convBlock(x, 256, strides=[2, 2])
    x = convBlock(x, 512, strides=[1, 1])
    x = convBlock(x, 512, strides=[2, 2])

    x = keras.layers.Dense(1024)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Flatten()(x)
    prob = keras.layers.Dense(1, activation='sigmoid')(x)

    return keras.models.Model(img, prob, name="Discriminator")

