import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import skimage
from skimage import io

"""
:returns lr_images, hr_images: tuple of tensors sized (N, 50, 50), (N, 500, 500)
"""


def get_data(hr_images_filepath, lr_image_width, hr_image_width):
    hr_images = []
    for i, hr_filename in enumerate(sorted(os.listdir(hr_images_filepath))[:1000]):  # TODO: Remove [:50] later
        if hr_filename.split('.')[-1] in {'TIF', 'tiff', 'tif'}:
            if i % 50 == 0:
                print("loading image", i + 0)
            print(os.path.join(hr_images_filepath, hr_filename))
            hr_image = skimage.img_as_float(skimage.io.imread(os.path.join(hr_images_filepath, hr_filename)))
            hr_image = np.where(hr_image > 255, 255, hr_image)
            hr_image = np.where(hr_image < 0, 0, hr_image)
            hr_images.append(hr_image)

    hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)  # takes longest

    print(hr_images.shape)
    # Normalize values within each channel
    hr_images_r = tf.expand_dims(tf.nn.softmax(hr_images[:, :, :, 0]), 3)
    hr_images_g = tf.expand_dims(tf.nn.softmax(hr_images[:, :, :, 1]), 3)
    hr_images_b = tf.expand_dims(tf.nn.softmax(hr_images[:, :, :, 2]), 3)
    hr_images = tf.concat([hr_images_r, hr_images_g, hr_images_b], axis=3)
    hr_images = tf.image.resize(hr_images, size=(hr_image_width, hr_image_width))  # TODO: remove later
    lr_images = tf.image.resize(hr_images, size=(lr_image_width, lr_image_width))
    train_size = tf.cast(lr_images.shape[0] * 0.75, tf.int32)
    return lr_images[:train_size], lr_images[train_size:], hr_images[:train_size], hr_images[train_size:]