from __future__ import absolute_import

from matplotlib import pyplot as plt

from CnnModel import CnnModel
from DownUpSample import DownUpSample
from preprocess import get_data

import tensorflow as tf

import numpy as np
from keras.applications.vgg16 import VGG16


def accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.))

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
        x = tf.keras.metrics.mean_squared_error(ori_high_res, pred_high_res)
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
        mse = tf.keras.metrics.mean_absolute_error(tf.reshape(ori_high_res, (ori_high_res, -1)),
                                                   tf.reshape(pred_high_res, (pred_high_res, -1)))
        psnr = tf.image.psnr(ori_high_res, pred_high_res, max_val=1.0)
        return psnr

    # label_images = tf.reshape(label_images, shape=(label_images.shape[0], -1))
    # predicted_images = tf.reshape(predicted_images, shape=(predicted_images.shape[0], -1))

    # predicted_images = np.where(predicted_images > 1., 1., predicted_images)
    # predicted_images = np.where(predicted_images < 0., 0., predicted_images)
    psnr = tf.reduce_mean(tf.image.psnr(label_images, predicted_images, max_val=1.))
    mse = tf.keras.metrics.mean_absolute_error(tf.reshape(label_images, (label_images.shape[0], -1)),
                                               tf.reshape(predicted_images, (predicted_images.shape[0], -1)))
    return -tf.reduce_mean(tf.image.psnr(label_images, predicted_images, max_val=1.))
    # return tf_ssim(label, predicted_image)
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - tf_ssim(label, predicted_image)) ## not working great
    # return 0.75 * tf.sqrt(mse(label, predicted_image)) + 0.25 * (1 - (1 + tf_ssim(label, predicted_image)) / 2)


def visualize_sr(input_images, predicted_images, train_labels, epoch, batch_num):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    difference = train_labels - predicted_images
    fig.suptitle("Visualizing SR for epoch " + str(epoch) + ", batch num " + str(batch_num))
    axs[0, 0].set_title('LR Input')
    axs[0, 1].set_title('Predicted HR Output')
    axs[1, 0].set_title('Actual HR Input')
    axs[1, 1].set_title('differences(ori-pred)')
    a = axs[0, 0].imshow(input_images[0] * 255, cmap='brg')
    plt.colorbar(a, ax=axs[0, 0])
    b = axs[0, 1].imshow(predicted_images[0] * 255, cmap='brg')
    plt.colorbar(b, ax=axs[0, 1])
    c = axs[1, 0].imshow(train_labels[0] * 255, cmap='brg')
    plt.colorbar(c, ax=axs[1, 0])
    d = axs[1, 1].imshow(difference[0] * 255, cmap='brg')
    plt.colorbar(d, ax=axs[1, 1])
    fig.tight_layout(pad=3)
    # plt.colorbar()
    plt.show()


def visualize_tst_sr(input_images, predicted_images, test_labels):
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    difference = test_labels - predicted_images
    fig.suptitle("Visualizing SR for epoch")
    plt.figure(figsize=(16, 12))
    axs[0, 0].set_title('LR Input')
    axs[0, 1].set_title('Predicted HR Output')
    axs[1, 0].set_title('Actual HR Input')
    axs[1, 1].set_title('differences(ori-pred)')
    a = axs[0, 0].imshow(input_images[0] * 255)
    plt.colorbar(a, ax=axs[0, 0])
    b = axs[0, 1].imshow(predicted_images[0] * 255)
    plt.colorbar(b, ax=axs[0, 1])
    c = axs[1, 0].imshow(test_labels[0] * 255)
    plt.colorbar(c, ax=axs[1, 0])
    d = axs[1, 1].imshow(difference[0] * 255)
    plt.colorbar(d, ax=axs[1, 1])
    fig.tight_layout(pad=3)
    plt.show()


def train(model, x, y_true, should_visualize_sr, epoch, batch_num):
    assert x.shape[0] == model.batch_size
    num_examples = tf.range(start=0, limit=model.batch_size)
    shuffle_indices = tf.random.shuffle(num_examples)
    x = tf.gather(x, shuffle_indices)
    y_true = tf.gather(y_true, shuffle_indices)
    with tf.GradientTape() as tape:  # init GT. model fwd prop monitored.
        y_pred = model.call(x)
        loss = loss_function(y_true, y_pred)
    print("Loss", loss)
    if should_visualize_sr:
        visualize_sr(x, y_pred, y_true, epoch, batch_num)
    x = model.trainable_variables
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, x, y_true):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.

    :param x: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param y_true: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    # preds = model.call(test_inputs, True)
    preds = model.call(x, True)
    accuracy = model.accuracy(preds, y_true)
    visualize_tst_sr(x, preds, y_true)
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
    lr_image_width, hr_image_width = 32, 128
    lr_train_images, lr_test_images, hr_train_images, hr_test_images = get_data(
        'H:\Shared drives\BU_DEMSuperRes_NW\super-resolution-dem\data\composite_database',
        lr_image_width, hr_image_width, 300)
    print('fetched images')
    # model = CnnModel(lr_image_width, hr_image_width)
    model = DownUpSample(lr_image_width, hr_image_width)
    print('model constructed')

    def get_batched(index, lr_images, hr_images):
        return lr_images[index:index + model.batch_size], hr_images[index:index + model.batch_size]

    NUM_EPOCHS = 100
    for ep in range(NUM_EPOCHS):
        # for each batch
        print('Epoch ', ep)
        for i in range(0, len(lr_train_images) - model.batch_size, model.batch_size):
            batched_lr_images, batched_hr_images = get_batched(i, lr_train_images, hr_train_images)
            train(model, batched_lr_images, batched_hr_images, should_visualize_sr=(i == 0), epoch=(ep + 1),
                  batch_num=i + 1)
            # visualize_loss(model.loss_list)
    accuracy = test(model, lr_test_images, hr_test_images)
    print(accuracy)


if __name__ == '__main__':
    main()
