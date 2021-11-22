import pickle
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import glob
import rasterio
import rasterio.features
import rasterio.warp


"""
Returns data: (N, H, W)

"""
def get_data(file_path):
    data = []
    for filename in os.listdir(file_path)[:50]:
        if filename.split('.')[-1] == 'TIF':
            print(filename)
            with rasterio.open(file_path + '/' + filename) as dataset:
                # Read the dataset's valid data mask as a ndarray.
                img = dataset.read()
                img = tf.reshape(img, shape=(img.shape[1], img.shape[2]))
                # slice mask
                img = img[:500,:500]
                print(img.shape)
                assert len(img) == 500 and len(img[0] == 500)
                data.append(img)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    # Normalize
    data = data / 255
    return data
