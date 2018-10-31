# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils

import os
import sys
import math
import tensorflow as tf

slim = tf.contrib.slim

# odd = 33001
# even = 31981

# for 0 and Others
# 0 : 6796
# others : 54971

_NUM_VALIDATION = 6796

_RANDOM_SEED = 0

_NUM_SHARDS = 6796

class ImageReader(object):

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  mnist_root = os.path.join(dataset_dir, 'images')
  mnist_root = dataset_dir
  photo_filenames = []
  print ('get file name and classes')
  for filename in os.listdir(mnist_root):
    path = os.path.join(mnist_root, filename)
    photo_filenames.append(path)

  return photo_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id, filename):
  base = os.path.basename(filename)
  temp = os.path.splitext(base)
  filename = temp[0]
  output_filename = 'mnist_eo_%s_%05d-of-%05d-name-%s.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS, filename)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['test']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, filenames[shard_id-1])

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i-1], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            example = dataset_utils.image_to_tfexample(
                image_data, b'png', height, width, 0)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['test']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def convert():
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """

  dataset_dir = '/home/sm/PycharmProjects/smproject/mnist_eo/testimages'
  # dataset_dir = '/home/sm/PycharmProjects/smproject/mnist_eo/dataset_eo/images/even'
  dataset_dir = '/home/sm/PycharmProjects/smproject/mnist_range_0andOthers/dataset_0andOthers/images/0'
  photo_filenames = _get_filenames_and_classes(dataset_dir)
  validation_filenames = photo_filenames

  _convert_dataset('test', validation_filenames, dataset_dir)

  print('\nFinished converting the Flowers dataset!')

  # Call test method
  test()

  # Delete validation file
  root = dataset_dir
  for filename1 in os.listdir(root):
      temp = filename1
      base = os.path.basename(filename1)
      base = os.path.splitext(base)
      expander = base[1]
      if (expander == ".tfrecord") :
          path = os.path.join(dataset_dir, temp)
          os.remove(path)

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

# CheckPoint Path for even vs odd
# tf.app.flags.DEFINE_string(
#     'checkpoint_path', '/home/sm/PycharmProjects/smproject/mnist_eo/train_eo',
#     'The directory where the model was written to or an absolute path to a '
#     'checkpoint file.')

# CheckPoint Path for 0 and others
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/sm/PycharmProjects/smproject/mnist_range_0andOthers/train_0andOthers',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'mnist_eo', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir','/home/sm/PycharmProjects/smproject/mnist_range_0andOthers/dataset_0andOthers/images/0', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'lenet', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'lenet', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def test():

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
      'mnist_eo', 'test', '/home/sm/PycharmProjects/smproject/mnist_range_0andOthers/dataset_0andOthers/images/0')

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=(2 - FLAGS.labels_offset),
      is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=False,
      common_queue_capacity=2 * 100,
      common_queue_min=100) # max number of test image!!!
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
        slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)

    test1 = []
    for i in range(100):
        test1.append(0)

    test2 = []
    for i in range(100):
        test2.append(1)

    # Define the metrics:

    # for even and odd
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #     'even': slim.metrics.streaming_accuracy(predictions, test1),
    #     'odd': slim.metrics.streaming_accuracy(predictions, test2),
    # })

    # for 0 and others
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        '0': slim.metrics.streaming_accuracy(predictions, test1),
        'others': slim.metrics.streaming_accuracy(predictions, test2),
    })

    # # Print Result
    summary_name = 'Result'
    op2 = tf.summary.scalar(summary_name, predictions[0], collections=[])
    op2 = tf.Print(op2, [predictions[0]], summary_name)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES, op2)

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.scalar_summary(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
        num_batches = FLAGS.max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)

if __name__ == '__main__':
    convert()