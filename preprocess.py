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


def get_data(hr_images_filepath, lr_image_width, hr_image_width, num_images=-1):
    hr_images = []
    for i, hr_filename in enumerate(sorted(os.listdir(hr_images_filepath))[:num_images]):
        if hr_filename.split('.')[-1] in {'TIF', 'tiff', 'tif'}:
            if i % 50 == 0:
                print("loading image", i + 0)
            print(os.path.join(hr_images_filepath, hr_filename))
            # load and sanitize image
            hr_image = skimage.img_as_float(skimage.io.imread(os.path.join(hr_images_filepath, hr_filename)))
            hr_image = np.where(hr_image > 255, 255, hr_image)
            hr_image = np.where(hr_image < 0, 0, hr_image)
            hr_image = tf.convert_to_tensor(hr_image)
            # Softmax 3 channels
            hr_image_r = tf.expand_dims(tf.nn.softmax(hr_image[:, :, 0]), 2)
            hr_image_g = tf.expand_dims(tf.nn.softmax(hr_image[:, :, 1]), 2)
            hr_image_b = tf.expand_dims(tf.nn.softmax(hr_image[:, :, 2]), 2)
            hr_image = tf.concat([hr_image_r, hr_image_g, hr_image_b], axis=2)
            hr_image = tf.image.resize(hr_image, size=(hr_image_width, hr_image_width))
            hr_images.append(hr_image)
    hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)  # takes longest
    lr_images = tf.image.resize(hr_images, size=(lr_image_width, lr_image_width))
    train_size = tf.cast(lr_images.shape[0] * 0.75, tf.int32)
    return lr_images[:train_size], lr_images[train_size:], hr_images[:train_size], hr_images[train_size:]
