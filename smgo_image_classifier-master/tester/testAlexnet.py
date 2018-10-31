#matplotlib inline

import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from nets import alexnet
from preprocessing import vgg_preprocessing

checkpoints_dir = '/home/sm/PycharmProjects/smproject/smgo/train_alexnet'

slim = tf.contrib.slim

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = alexnet.alexnet_v2.default_image_size

with tf.Graph().as_default():
    image = open('/home/sm/PycharmProjects/smproject/smgo/temp/images.jpeg', 'rb')

    # Open specified url and load image as a string
    image_string = image.read()

    # Decode string into matrix with intensity values
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    processed_image = vgg_preprocessing.preprocess_image(image,
                                                         image_size,
                                                         image_size,
                                                         is_training=False)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images = tf.expand_dims(processed_image, 0)

    # Create the model, use the default arg scope to configure
    # the batch norm parameters. arg_scope is a very conveniet
    # feature of slim library -- you can define default
    # parameters for layers -- like stride, padding etc.
    with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
        logits, _ = alexnet.alexnet_v2(processed_images,
                                       num_classes=4,
                                       is_training=False)

    # In order to get probabilities we apply softmax on the output.
    probabilities = tf.nn.softmax(logits)

    # Create a function that reads the network weights
    # from the checkpoint file that you downloaded.
    # We will run it in session later.
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'model.ckpt-454123'),
        slim.get_model_variables('alexnet_v2'))

    with tf.Session() as sess:
        # Load weights
        init_fn(sess)

        # We want to get predictions, image as numpy matrix
        # and resized and cropped piece that is actually
        # being fed to the network.
        np_image, network_input, probabilities = sess.run([image,
                                                           processed_image,
                                                           probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                            key=lambda x: x[1])]

    # Show the downloaded image
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    # Show the image that is actually being fed to the network
    # The image was resized while preserving aspect ratio and then
    # cropped. After that, the mean pixel value was subtracted from
    # each pixel of that crop. We normalize the image to be between [-1, 1]
    # to show the image.
    plt.imshow(network_input / (network_input.max() - network_input.min()))
    plt.suptitle("Resized, Cropped and Mean-Centered input to network",
                 fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

    names = ['faces', 'food', 'nature', 'pets']
    for i in range(4):
        index = sorted_inds[i]
        print('Probability %0.2f => [%s]' % (probabilities[index], names[index]))

    # if probability < 0.5, save into ETC folder
    # if (probabilities[sorted_inds[0]] < 0.5f):


    res = slim.get_model_variables()