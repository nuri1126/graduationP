
import os
import tensorflow as tf

from nets import lenet
from preprocessing import lenet_preprocessing

checkpoints_dir = '/home/sm/PycharmProjects/smproject/mnist_11/train_11'

slim = tf.contrib.slim

# We need default size of image for a particular network.
image_size = lenet.lenet.default_image_size

names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'nature_png']

def classify_image(filepath):
    with tf.Graph().as_default():
        image = open(filepath, 'rb')

        # Open specified url and load image as a string
        image_string = image.read()

        # Decode string into matrix with intensity values
        image = tf.image.decode_png(image_string, channels=3)

        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.
        processed_image = lenet_preprocessing.preprocess_image(image,
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
        with slim.arg_scope(lenet.lenet_arg_scope()):
            logits, _ = lenet.lenet(processed_images,
                                   num_classes=11,
                                   is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'model.ckpt-10000'),
            slim.get_model_variables(None))

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

        for i in range(11):
            index = sorted_inds[i]
            print('Probability %0.2f => [%s]' % (probabilities[index], names[index]))

    return sorted_inds[0], probabilities
