import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, LeakyReLU, Concatenate, Conv2DTranspose, Input


class DownUpSample(tf.keras.Model):
    def __init__(self, lr_image_width, hr_image_width):
        if hr_image_width % lr_image_width != 0:
            raise ValueError("hr_image_width must be divisible by lr_image_width")
        self.upscale_factor = hr_image_width // lr_image_width
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(DownUpSample, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.batch_size = 56
        self.W = lr_image_width  # img width
        self.H = lr_image_width  # img height

        class Up(tf.keras.Model):
            def __init__(self, filters, kernel_size, dropout=False):
                super(Up, self).__init__()
                self.W = lr_image_width
                self.H = lr_image_width
                self.dropout = dropout
                self.model1 = keras.Sequential(
                    [
                        Conv2DTranspose(filters, kernel_size, padding='same', strides=2),
                        Dropout(.2),
                    ]
                )
                self.model2 = keras.Sequential(
                    [
                        LeakyReLU()
                    ]
                )

            def call(self, inputs):
                x = self.model1(inputs)
                if self.dropout:
                    x = Dropout(.2)(x)
                x = self.model2(x)
                return x

        class Down(tf.keras.Model):
            def __init__(self, filters, kernel_size, apply_batch_normalization=True):
                super(Down, self).__init__()
                self.W = lr_image_width
                self.H = lr_image_width
                self.apply_batch_normalization = apply_batch_normalization
                self.model1 = keras.Sequential(
                    [
                        Conv2D(filters, kernel_size, padding='same', strides=2),
                        Dropout(.2),
                    ]
                )
                self.model2 = keras.Sequential(
                    [
                        LeakyReLU()
                    ]
                )

            def call(self, inputs):
                x = self.model1(inputs)
                if self.apply_batch_normalization:
                    x = BatchNormalization()(x)
                x = self.model2(x)
                return x

        self.d1 = Down(128, (3, 3), False)
        self.d2 = Down(128, (3, 3), False)
        self.d3 = Down(256, (3, 3), True)
        self.d4 = Down(512, (3, 3), True)
        self.d5 = Down(512, (3, 3), True)
        # upsampling
        self.u1 = Up(512, (3, 3), False)
        self.u2 = Up(256, (3, 3), False)
        self.u3 = Up(128, (3, 3), False)
        self.u4 = Up(128, (3, 3), False)
        self.u5 = Up(3, (3, 3), False)
        self.enlarged = Up(1, (3, 3), False)
        self.enlarged2 = Up(1, (3, 3), False)
        self.out = Conv2D(3, (2, 2), strides=1, padding='same')

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # down
        d1 = self.d1.call(inputs)
        d2 = self.d2.call(d1)
        d3 = self.d3.call(d2)
        # up
        u1 = self.u1.call(d3)
        u1 = tf.concat([u1, d2], -1)
        u2 = self.u2.call(u1)
        u2 = tf.concat([u2, d1], -1)
        u3 = self.u3.call(u2)
        u4 = self.u4.call(u3)
        u5 = self.u5.call(u4)
        out = self.out(u5)
        return out
