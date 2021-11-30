from __future__ import absolute_import

from matplotlib import pyplot as plt

from CnnModel import CnnModel
from preprocess import get_data

import tensorflow as tf

import numpy as np
from keras.applications.vgg16 import VGG16


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


def loss_function(label_images, predicted_images):
    def tf_ssim_multiscale(ori_high_res, pred_high_res):
        ori_high_res = tf.expand_dims(ori_high_res, axis=3)
        pred_high_res = tf.expand_dims(pred_high_res, axis=3)
        print('start')
        print('tf.image.ssim_multiscale(ori_high_res, pred_high_res, max_val=1.)',
              tf.image.ssim_multiscale(ori_high_res, pred_high_res, max_val=1.))
        return tf.reduce_mean(tf.image.ssim_multiscale(ori_high_res, pred_high_res, max_val=1.))

    def tf_ssim(ori_high_res, pred_high_res):
        return tf.reduce_mean(tf.image.ssim(ori_high_res, pred_high_res, max_val=1.))

    def mse(ori_high_res, pred_high_res):
        '''
        Mean square error
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

    return 0.0 * (1 - (1 + tf_ssim(label_images, predicted_images)) / 2) + 1.0 * tf.sqrt(
        mse(label_images, predicted_images))
    # return .25 * (1 - (1 + tf_ssim(label_images, predicted_images)) / 2) + .75 * tf.sqrt(mse(label_images, predicted_images))
    # return tf_ssim(label, predicted_image)
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - tf_ssim(label, predicted_image)) ## not working great
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - (1 + tf_ssim(label, predicted_image)) / 2)


def visualize_sr(input_images, predicted_images, train_labels, epoch, batch_num):
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Visualizing SR for epoch " + str(epoch) + ", batch num " + str(batch_num))
    axs[0].imshow(input_images[0], cmap='gray')
    axs[0].set_title('LR Input')
    axs[1].imshow(predicted_images[0], cmap='gray')
    axs[1].set_title('Predicted HR Output')
    axs[2].imshow(train_labels[0], cmap='gray')
    axs[2].set_title('Actual HR Input')
    plt.show()


def train(model, train_inputs, train_labels, should_visualize_sr, epoch, batch_num):
    assert train_inputs.shape[0] == model.batch_size
    num_examples = tf.range(start=0, limit=model.batch_size)
    shuffle_indices = tf.random.shuffle(num_examples)
    train_inputs = tf.gather(train_inputs, shuffle_indices)
    train_labels = tf.gather(train_labels, shuffle_indices)
    with tf.GradientTape() as tape:  # init GT. model fwd prop monitored.
        predicted_images = model.call(train_inputs)
        loss = loss_function(train_labels, predicted_images)
    print("Loss", loss)
    if should_visualize_sr:
        visualize_sr(train_inputs, predicted_images, train_labels, epoch, batch_num)
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


def main():
    # Read in Arctic DEM data
    lr_image_width, hr_image_width = 50, 100
    lr_train_images, lr_test_images, hr_train_images, hr_test_images = get_data('data/arctic_dem_2m_10000_imgs',
                                                                                lr_image_width, hr_image_width)

    model = CnnModel(lr_image_width, hr_image_width)

    def get_batched(index, lr_images, hr_images):
        return lr_images[index:index + model.batch_size], hr_images[index:index + model.batch_size]

    NUM_EPOCHS = 100
    for ep in range(NUM_EPOCHS):
        # for each batch
        print('Epoch ', ep)
        for i in range(0, len(lr_train_images) - model.batch_size, model.batch_size):
            batched_lr_images, batched_hr_images = get_batched(i, lr_train_images, hr_train_images)
            train(model, batched_lr_images, batched_hr_images, should_visualize_sr=(i % 10 == 0), epoch=(ep + 1),
                  batch_num=i + 1)
            # visualize_loss(model.loss_list)


if __name__ == '__main__':
    main()
