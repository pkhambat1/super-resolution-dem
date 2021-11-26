from __future__ import absolute_import
from _typeshed import Self

import os

os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
from matplotlib import pyplot as plt

from preprocess import get_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D
import numpy as np
import math


class Model(tf.keras.Model):
    def __init__(self, ):
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.batch_size = 126
        self.W = 500  # img width
        self.H = 500  # img height
        self.num_classes = 2
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main

        self.feed_forward = keras.Sequential(
            [
                Flatten(),
                # keras.Input(shape=(self.W * self.H)),
                # Conv2D(64, 3, strides=(1, 1), padding='same'),
                # layers.BatchNormalization(),
                # layers.LeakyReLU(),
                # layers.Conv2D(64, 3, strides=(1, 1), padding='same'),
                # layers.BatchNormalization(),
                # layers.LeakyReLU(),
            ]
        )

        # TODO: Initialize all trainable parameters

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        X = self.feed_forward(inputs)
        print(self.feed_forward.summary())
        print(X.shape)
        return X

    def tf_fspecial_gauss(self,size, sigma):
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


    def tf_ssim(self, ori_high_res, pred_high_res, cs_map=False, mean_metric=True, size=11, sigma=1.5):
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
        window = self.tf_fspecial_gauss(size, sigma)  # window shape [size, size]
        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = tf.nn.conv2d(ori_high_res, window, strides=[1, 1, 1, 1], padding='VALID')
        mu2 = tf.nn.conv2d(pred_high_res, window, strides=[1, 1, 1, 1], padding='VALID')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = tf.nn.conv2d(ori_high_res * ori_high_res, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(pred_high_res * pred_high_res, window, strides=[1, 1, 1, 1], padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(ori_high_res * pred_high_res, window, strides=[1, 1, 1, 1], padding='VALID') - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2)),
                    (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value


    @DeprecationWarning
    def tf_ms_ssim(ori_high_res, pred_high_res, mean_metric=True, level=5):
        """
        Compute multi-scale structural similarity index metric.
        https://stackoverflow.com/questions/39051451/ssim-ms-ssim-for-tensorflow

        :param img1:
        :param img2:
        :param mean_metric:
        :param level:
        :return: msssim
        """
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []
        for l in range(level):
            ssim_map, cs_map = tf_ssim(ori_high_res, pred_high_res, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            filtered_im1 = tf.nn.avg_pool(ori_high_res, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(pred_high_res, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            ori_high_res = filtered_im1
            pred_high_res = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level - 1] ** weight[0:level - 1]) *
                (mssim[level - 1] ** weight[level - 1]))

        if mean_metric:
            value = tf.reduce_mean(value)
        return value

    def mse(ori_high_res,pred_high_res):
        '''
        Mean squre error
        '''
        return tf.keras.metrics.mean_squared_error(ori_high_res, pred_high_res)

    def mae(ori_high_res,pred_high_res):
        '''
        mean absolute error
        '''
        mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM)
        return mae(ori_high_res, pred_high_res).numpy()



    def tf_psnr(ori_high_res,pred_high_res):
        """
        PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

        Adopted from "https://www.tensorflow.org/api_docs/python/tf/image/psnr"

        """
        psnr=tf.image.psnr(ori_high_res, pred_high_res, max_val=255, name=None)
        return psnr

    
    def accuracy(self, logits, labels):
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


def train(model, train_inputs, train_labels, mode="mae"):
    train_inputs = tf.image.random_flip_left_right(train_inputs)
    # Implement backprop:
    with tf.GradientTape() as tape:  # init GT. model fwd prop monitored.
        predicted_image = model.call(train_inputs) 
         # this calls the call function conveniently
        if mode == "mae":
            loss = model.mae(train_inputs,predicted_image)
        elif mode == "psnr":
            loss=model.psnr(train_inputs,predicted_image)
        elif mode=="mse":
            loss=model.mse(train_inputs,predicted_image)
        elif mode=="tf_ssim":
            loss=model.tf_ssim(train_inputs,predicted_image)
        elif mode=="tf_ms_ssim":
            loss=model.tf_ms_ssim(train_inputs,predicted_image)
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
    lr_images, hr_images = get_data('H:\Shared drives\BU_DEMSuperRes_NW\super-resolution-dem\data\ArcticDEM_20m_lr', 'H:\Shared drives\BU_DEMSuperRes_NW\super-resolution-dem\data\ArcticDEM_2m_hr')
    model = Model()

    def get_batched(index, lr_images, hr_images):
        return lr_images[index:index + model.batch_size], hr_images[index:index + model.batch_size]

    for _ in range(10):
        # for each batch
        for i in range(0, len(lr_images), model.batch_size):
            batched_lr_images, batched_hr_images = get_batched(i, lr_images, hr_images)
            train(model, batched_lr_images, batched_hr_images)

    return None


if __name__ == '__main__':
    main()
