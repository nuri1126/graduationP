from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import os

import tensorflow as tf

from tester_mnist_11 import mnist_11_nature_classifier

slim = tf.contrib.slim

dataset_dir = '/home/sm/PycharmProjects/smproject/mnist_11/testimages'

def main():
    photo_filenames = _get_filenames(dataset_dir)

    for i in range(photo_filenames.__len__()):
        print('####################################################################', i+1)
        temp = photo_filenames[i]
        base = os.path.basename(temp)
        print('>> File name :', base)
        mnist_11_nature_classifier.classify_image(photo_filenames[i])

def _get_filenames(dataset_dir):
  smgo_root = os.path.join(dataset_dir, 'images')
  photo_filenames = []
  for filename in os.listdir(smgo_root):
    path = os.path.join(smgo_root, filename)
    photo_filenames.append(path)
  return photo_filenames

if __name__ == '__main__':
    main()