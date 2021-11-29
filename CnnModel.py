import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Reshape, Conv2D, Input, Lambda


class CnnModel(tf.keras.Model):
    def __init__(self, upscale_factor):
        self.upscale_factor = int(upscale_factor)
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CnnModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.batch_size = 54
        self.W = 50  # img width
        self.H = 50  # img height
        self.num_classes = 2
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main
        self.conv_args = {
            "activation": "relu",
            "kernel_initializer": "Orthogonal",
            "padding": "same",
        }
        # self.vgg19 = tf.keras.applications.vgg19.VGG19(input_shape=(500, 500, 1), include_top=False)
        self.feed_forward = keras.Sequential(
            [
                Reshape(target_shape=(self.W, self.H, 1)),
                Input(shape=(self.W, self.H, 1)),
                Conv2D(64, kernel_size=5, **self.conv_args),
                Conv2D(64, kernel_size=3, **self.conv_args),
                # Conv2D(64, kernel_size=3, padding="same", activation='relu'),
                # Conv2D(64, kernel_size=3, padding="same",activation='relu'),
                # Conv2D(64, kernel_size=3, padding="same", activation='relu'),
                Conv2D(32, kernel_size=3, **self.conv_args),
                Conv2D(self.upscale_factor ** 2, kernel_size=3, **self.conv_args),
                Lambda(lambda x: tf.nn.depth_to_space(x, self.upscale_factor)),
                # Lambda(lambda x: self.vgg19(x)),
                Reshape(target_shape=(self.W * self.upscale_factor, self.H * self.upscale_factor))
            ]
        )

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        return self.feed_forward(inputs)
