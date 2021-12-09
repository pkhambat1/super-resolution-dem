import math

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Concatenate, Conv2DTranspose, Input

from PreUpsampleCnn import PreUpsampleCnn

class ProgressiveUpsampleCnn(tf.keras.Model):
    '''
    Observation: Images turning out very 3D looking and colors are inaccurate compared to Vanilla CNN
    '''

    def __init__(self, lr_image_width, hr_image_width, lr_pretrain_images, hr_pretrain_images):
        # if hr_image_width % lr_image_width != 0:
        #     raise ValueError("hr_image_width must be divisible by lr_image_width")
        super(ProgressiveUpsampleCnn, self).__init__()
        upscale_factor = hr_image_width // lr_image_width
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.batch_size = 56
        intermediate_width = lr_image_width * int(math.sqrt(hr_image_width // lr_image_width))
        self.du1 = PreUpsampleCnn(lr_image_width, intermediate_width)
        self.du2 = PreUpsampleCnn(intermediate_width, hr_image_width)

    def call(self, inputs):
        x = self.du1(inputs)
        x = self.du2(x)
        return x
