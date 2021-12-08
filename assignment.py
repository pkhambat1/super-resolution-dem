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

def loss_function(y_pred, y_true):
    def total_variation_loss(image):
        '''
        To reduce image noise - https://medium.com/@shwetaka1988/a-complete-step-wise-guide-on-neural-style-transfer-9f60b22b4f75
        :param image:
        :return:
        '''

        def high_pass_x_y(image):
            x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
            y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
            return x_var, y_var

        x_deltas, y_deltas = high_pass_x_y(image)
        return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)

    def get_activation_map(images, conv_layer='block1_conv1', should_visualize=False):
        images = tf.image.resize(images, (224, 224))
        layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        # layer_names = ['block1_conv1']
        intermediate_layer_model = tf.keras.Model(inputs=vgg19.input,
                                                  outputs=[vgg19.get_layer(conv_layer).output for conv_layer in
                                                           layer_names])
        intermediate_outputs = intermediate_layer_model(images)
        if should_visualize:
            for intermediate_output in intermediate_outputs:
                fig, axes = plt.subplots(8, 8, figsize=(12, 12))
                for i in range(64):
                    for j in range(images.shape[0]):
                        axes[int(i / 8), i % 8].imshow(intermediate_output[j, :, :, i])
                plt.show()
        return intermediate_outputs

    # y_pred_features_list = get_activation_map(y_pred)
    # y_true_features_list = get_activation_map(y_true)
    # var_loss = total_variation_loss(y_pred)
    # content_loss = tf.reduce_mean(
    #     [-tf.reduce_mean(tf.image.psnr(y_pred_feature, y_true_feature, max_val=1.)) for y_pred_feature, y_true_feature
    #      in zip(y_pred_features_list, y_true_features_list)])
    image_loss = -tf.reduce_mean(tf.image.psnr(y_pred, y_true, max_val=1.))
    # return .50 * image_loss + .25 * content_loss + .25 * var_loss
    return image_loss


# def visualize_sr(input_images, predicted_images, train_labels, epoch, batch_num):
#     fig, axs = plt.subplots(1, 3)
#     fig.suptitle("Visualizing SR for epoch " + str(epoch) + ", batch num " + str(batch_num))
#     axs[0].set_title('LR Input')
#     axs[1].set_title('Predicted HR Output')
#     axs[2].set_title('Actual HR Input')
#     axs[0].imshow(input_images[0] * 255)
#     axs[1].imshow(predicted_images[0] * 255)
#     axs[2].imshow(train_labels[0] * 255)
#     plt.show()


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
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


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
