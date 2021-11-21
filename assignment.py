from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class Model(tf.keras.Model):
    def __init__(self, ):
        """
        This model class will contain the architecture for your CNN that
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        self.batch_size = 100
        self.num_classes = 2
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # TODO: Initialize all hyperparameters
        self.conv_w1 = tf.Variable(tf.random.truncated_normal([5, 5, 3, 16], stddev=.1))
        self.conv_w2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 32], stddev=.1))
        self.conv_w3 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=.1))
        self.conv_w4 = tf.Variable(tf.random.truncated_normal([5, 5, 64, 64], stddev=.1))
        self.conv_w5 = tf.Variable(tf.random.truncated_normal([3, 3, 64, 128], stddev=.1))
        self.conv_w6 = tf.Variable(tf.random.truncated_normal([3, 3, 128, 128], stddev=.1))
        self.conv_w7 = tf.Variable(tf.random.truncated_normal([3, 3, 128, 128], stddev=.1))

        self.conv_b1 = tf.Variable(tf.random.truncated_normal([16], stddev=.1))
        self.conv_b2 = tf.Variable(tf.random.truncated_normal([32], stddev=.1))
        self.conv_b3 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))
        self.conv_b4 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))
        self.conv_b5 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))
        self.conv_b6 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))
        self.conv_b7 = tf.Variable(tf.random.truncated_normal([128], stddev=.1))

        self.W1 = tf.Variable(
            tf.random.truncated_normal([128, 64],
                                       stddev=.1))  # input dim, output dim. input dim should be output of flattened lyr
        self.b1 = tf.Variable(tf.random.truncated_normal([64], stddev=.1))
        self.W2 = tf.Variable(tf.random.truncated_normal([64, 32], stddev=.1))
        self.b2 = tf.Variable(tf.random.truncated_normal([32], stddev=.1))
        self.W3 = tf.Variable(tf.random.truncated_normal([32, 2], stddev=.1))
        self.b3 = tf.Variable(tf.random.truncated_normal([2], stddev=.1))

        # TODO: Initialize all trainable parameters

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        strides1 = [1, 2, 2, 1]
        ksize1 = [1, 3, 3, 1]
        ksize2 = [1, 2, 2, 1]
        eps = 1e-5

        # non-dense layer
        # Convolution Layer 1
        X = tf.nn.conv2d(inputs, self.conv_w1, strides=strides1, padding='SAME')
        X = tf.nn.bias_add(X, self.conv_b1)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, padding='SAME', strides=strides1, ksize=ksize1)

        # Convolution Layer 2
        X = tf.nn.conv2d(X, self.conv_w2, strides=strides1, padding='SAME')
        X = tf.nn.bias_add(X, self.conv_b2)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, padding='SAME', strides=strides1, ksize=ksize2)

        # Convolution Layer 3
        if is_testing:
            X = conv2d(X, self.conv_w3, strides=[1, 1, 1, 1], padding='SAME')
        else:
            X = tf.nn.conv2d(X, self.conv_w3, strides=[1, 1, 1, 1], padding='SAME')

        X = tf.nn.bias_add(X, self.conv_b3)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, padding='SAME', strides=strides1, ksize=ksize2)

        # Convolution Layer 4
        X = tf.nn.conv2d(X, self.conv_w4, strides=strides1, padding='SAME')
        X = tf.nn.bias_add(X, self.conv_b4)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, padding='SAME', strides=strides1, ksize=ksize2)

        # Convolution Layer 5
        X = tf.nn.conv2d(X, self.conv_w5, strides=strides1, padding='SAME')
        X = tf.nn.bias_add(X, self.conv_b5)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)
        X = tf.nn.max_pool(X, padding='SAME', strides=strides1, ksize=ksize2)

        # Convolution Layer 6
        X = tf.nn.conv2d(X, self.conv_w6, strides=strides1, padding='SAME')
        X = tf.nn.bias_add(X, self.conv_b6)
        tf.nn.batch_normalization(X, mean=tf.nn.moments(X, axes=[0, 1, 2])[0],
                                  variance=tf.nn.moments(X, axes=[0, 1, 2])[1], variance_epsilon=eps, offset=0,
                                  scale=1)
        X = tf.nn.relu(X)

        # flatten to rm 2d relationships
        X = tf.reshape(X, [X.shape[0], -1])  # (100, 80)

        # try Relu on Dense layers and dropout in conv layers
        # try data aug, ....
        # Dense layer 1
        X = tf.matmul(X, self.W1) + self.b1  # (100, 80) * (80, 40) = (100, 40)
        X = tf.nn.relu(X)
        X = tf.nn.dropout(X, rate=0.05)

        # Dense layer 2
        X = tf.matmul(X, self.W2) + self.b2  # (100, 40) * (40, 20) = (100, 20)
        X = tf.nn.relu(X)
        X = tf.nn.dropout(X, rate=0.05)

        # Dense layer 3
        X = tf.matmul(X, self.W3) + self.b3  # (100, 20) * (20, 2) = (100, 2)
        return X

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

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


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    # assert not np.array_equal(train_inputs, tf.image.random_flip_left_right(train_inputs))
    train_inputs = tf.image.random_flip_left_right(train_inputs)
    # Implement backprop:
    with tf.GradientTape() as tape:  # init GT. model fwd prop monitored.
        predictions = model(train_inputs)  # this calls the call function conveniently
        loss = model.loss(predictions, train_labels)
        model.loss_list.append(np.average(loss))
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    accuracy = model.accuracy(predictions, train_labels)
    print(accuracy)


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
    images = get_data('data/Spllited_without_small_file_2m')


if __name__ == '__main__':
    main()
