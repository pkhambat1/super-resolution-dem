from __future__ import absolute_import
# from _typeshed import Self

import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt

from preprocess import get_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, ReLU, Input
import numpy as np
import math


class Model(tf.keras.Model):
    def __init__(self, upscale_factor):
        self.upscale_factor = int(upscale_factor)
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.batch_size = 128
        self.W = 50  # img width
        self.H = 50  # img height
        self.num_classes = 2
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main
        self.feed_forward = keras.Sequential(
            [
                Reshape(target_shape=(self.W, self.H, 1)),
                Input(shape=(self.W, self.H, 1)),
                Conv2D(64, kernel_size=5, padding="same", activation='relu'),
                Conv2D(64, kernel_size=3, padding="same", activation='relu'),
                Conv2D(64, kernel_size=3, padding="same", activation='relu'),
                Conv2D(64, kernel_size=3, padding="same",activation='relu'),
                Conv2D(64, kernel_size=3, padding="same", activation='relu'),
                Conv2D(32, kernel_size=3, padding="same", activation='relu'),
                Conv2D(int(self.upscale_factor ** 2), kernel_size=3, padding="same", activation='relu'),
            ]
        )
        self.conv5 = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same")
        # TODO: Initialize all trainable parameters

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        out = self.feed_forward(inputs)
        self.feed_forward.summary()
        out = tf.nn.depth_to_space(out, self.upscale_factor)
        # out = self.conv5(out)
        out = tf.reshape(out, shape=(-1, 500, 500))
        return out

def accuracy(logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def loss_function(label, predicted_image):


    def tf_ssim(ori_high_res, pred_high_res, cs_map=False, mean_metric=True, size=11, sigma=1.5):
        """
        Compute structural similarity index metric.
        https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow

        :param img1: an input image
        :param img2: an input image
        :param cs_map:
        :param mean_metric:
        :param size:
        :param sigma:
        :return: ssim
        """
        ssim = tf.image.ssim(ori_high_res, pred_high_res, max_val=1.0)
        return tf.reduce_sum(ssim)

    def mse(ori_high_res, pred_high_res):
        '''
        Mean squre error
        '''
        return tf.reduce_mean(tf.keras.metrics.mean_squared_error(ori_high_res, pred_high_res))

    def tf_fspecial_gauss(size, sigma):
        """Function to mimic the 'fspecial' gaussian MATLAB function
        :param size:
        :param sigma:
        :return:
        """
        x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

        x_data = np.expand_dims(x_data, axis=-1)
        x_data = np.expand_dims(x_data, axis=-1)

        y_data = np.expand_dims(y_data, axis=-1)
        y_data = np.expand_dims(y_data, axis=-1)

        x = tf.constant(x_data, dtype=tf.float32)
        y = tf.constant(y_data, dtype=tf.float32)

        g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / tf.reduce_sum(g)

    def mae(ori_high_res, pred_high_res):
        '''
        mean absolute error
        '''
        # mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.MEAN)
        # #print(mae)
        # mean_ae=tf.reduce_mean(mae(ori_high_res, pred_high_res).numpy())
        # print (mean_ae)
        mae = tf.keras.metrics.mean_absolute_error(ori_high_res, pred_high_res)
        return tf.reduce_mean(mae)

    def tf_psnr(ori_high_res, pred_high_res):
        """
        PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

        Adopted from "https://www.tensorflow.org/api_docs/python/tf/image/psnr"

        """
        psnr = tf.image.psnr(ori_high_res, pred_high_res, max_val=1.0, name=None)
        return psnr

    return mae(label, predicted_image)
    # return tf.sqrt(mse(label, predicted_image))
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - tf_ssim(label, predicted_image))
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - (1 + tf_ssim(label, predicted_image)) / 2)

def train(model, train_inputs, train_labels):
    assert train_inputs.shape[0] == model.batch_size
    num_examples = tf.range(start=0, limit=model.batch_size, dtype=tf.int32)
    shuffle_indices = tf.random.shuffle(num_examples)
    train_data = tf.gather(train_inputs, shuffle_indices)
    label_images = tf.gather(train_labels, shuffle_indices)
    with tf.GradientTape() as tape:  # init GT. model fwd prop monitored.
        predicted_images = model.call(train_data)
        _, axs = plt.subplots(2)
        # fig.suptitle('Predicted (L), Ground Truth (R)')
        axs[0].imshow(predicted_images[0])
        axs[1].imshow(label_images[0])
        loss = loss_function(label_images, predicted_images)
    model.loss_list.append(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    # preds = model.call(test_inputs, True)
    preds = model.call(test_inputs, False)
    accuracy = model.accuracy(preds, test_labels)
    return accuracy


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """

    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i + 1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    # Read in Arctic DEM data
    lr_images, hr_images = get_data('data/ArcticDEM_20m_lr', 'data/ArcticDEM_2m_hr')
    model = Model(upscale_factor=10)

    def get_batched(index, lr_images, hr_images):
        return lr_images[index:index + model.batch_size], hr_images[index:index + model.batch_size]

    NUM_EPOCHS = 3
    for ep in range(NUM_EPOCHS):
        # for each batch
        for i in range(0, len(lr_images) - model.batch_size, model.batch_size):
            print('Epoch ', ep)
            batched_lr_images, batched_hr_images = get_batched(i, lr_images, hr_images)
            train(model, batched_lr_images, batched_hr_images)
            visualize_loss(model.loss_list)
    return None


if __name__ == '__main__':
    main()
