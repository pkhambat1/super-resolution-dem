import pickle
import numpy as np
import tensorflow as tf
import os
from tifffile import tifffile
from sklearn.model_selection import train_test_split


"""
:returns lr_images, hr_images: tuple of tensors sized (N, 50, 50), (N, 500, 500)
"""
def get_data(lr_images_filepath, hr_images_filepath):
    lr_images, hr_images = [], []
    assert len(sorted(os.listdir(lr_images_filepath))) == len(sorted(os.listdir(hr_images_filepath)))
    for lr_filename, hr_filename in zip(sorted(os.listdir(lr_images_filepath))[:500], sorted(os.listdir(hr_images_filepath))): # TODO: Remove [:50] later
        if lr_filename.split('.')[-1] in {'TIF', 'tiff'}:
            assert lr_filename == hr_filename # essential to map LR to HR correctly
            lr_image = tifffile.imread(os.path.join(lr_images_filepath, lr_filename))
            hr_image = tifffile.imread(os.path.join(hr_images_filepath, hr_filename))
            # eliminate off values
            lr_image = np.where(lr_image >= 255, 255/2, lr_image)
            hr_image = np.where(hr_image >= 255, 255/2, hr_image)
            lr_images.append(lr_image)
            hr_images.append(hr_image)

    lr_images = tf.convert_to_tensor(lr_images, dtype=tf.float32)
    hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)
    # Normalize
    lr_images /= 255
    hr_images /= 255
    train_size = tf.cast(lr_images.shape[0] * 0.75, tf.int32)
    return lr_images[:train_size], lr_images[train_size:], hr_images[:train_size], hr_images[train_size:]
