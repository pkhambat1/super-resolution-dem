import pickle
import numpy as np
import tensorflow as tf
import os
from tifffile import tifffile


"""
Returns data: (N, H, W)

"""
def get_data(filepath):
    data = []
    for filename in os.listdir(filepath):
        if filename.split('.')[-1] in {'TIF', 'tiff'}:
            image = tifffile.imread(os.path.join(filepath, filename))
            # image = tifffile.imread(os.path.join(filepath, '..', '..', 'dummy_data', 'dog.tiff')) # TODO: delete later
            image = image[:500,:500]
            # eliminate off values
            image = np.where(image >= 255, 0, image)
            data.append(image)

    data = tf.convert_to_tensor(data, dtype=tf.float32)
    # # Normalize
    data = data / 255
    return data
