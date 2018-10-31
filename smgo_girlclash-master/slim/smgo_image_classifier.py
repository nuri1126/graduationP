
import os
import tensorflow as tf

from slim import alexnet
from slim import vgg_preprocessing

checkpoints_dir = '/home/sm/PycharmProjects/test/slim/model_alexnet'

slim = tf.contrib.slim

# We need default size of image for a particular network.
image_size = alexnet.alexnet_v2.default_image_size

names = ['faces', 'fashion', 'food', 'nature', 'pets']

def classify_image(filepath):
    with tf.Graph().as_default():
        image = open(filepath, 'rb')

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
        # the batch norm parameters. arg_scope is a very convenient
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            logits, _ = alexnet.alexnet_v2(processed_images,
                                   num_classes=5,
                                   is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt-100000'),
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

    return sorted_inds[0], probabilities
