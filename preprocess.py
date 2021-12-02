import pickle
import numpy as np
import tensorflow as tf
import os
from tifffile import tifffile
from sklearn.model_selection import train_test_split

"""
:returns lr_images, hr_images: tuple of tensors sized (N, 50, 50), (N, 500, 500)
"""


def get_data(hr_images_filepath, lr_image_width, hr_image_width):
    hr_images = []
    for hr_filename in os.listdir(hr_images_filepath):  # TODO: Remove [:50] later
        if hr_filename.split('.')[-1] in {'TIF', 'tiff','tif'}:
            hr_image = tifffile.imread(os.path.join(hr_images_filepath, hr_filename))
            hr_image = np.where(hr_image >= 255, 255 / 2, hr_image)
            hr_images.append(hr_image)

    hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)
    # Normalize
    hr_images /= 255
    print (hr_images.shape)
    hr_images = tf.image.resize(hr_images, size=(hr_image_width,hr_image_width))# TODO: remove later
    lr_images = tf.image.resize(hr_images, size=(lr_image_width,lr_image_width))
    train_size = tf.cast(lr_images.shape[0] * 0.75, tf.int32)
    return lr_images[:train_size], lr_images[train_size:], hr_images[:train_size], hr_images[train_size:]
