from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import division
from __future__ import print_function
from __future__ import print_function

import os
import shutil

import tensorflow as tf

from tester_smgo_with_garbage import smgo_with_garbage_classifier

slim = tf.contrib.slim

dataset_dir = '/home/sm/PycharmProjects/smproject/smgo/testimages'

classes = ['_faces_6', '_fashion_6', '_food_6', '_etc_6', '_nature_6', '_pets_6']

MIN_PROBABIL = 0.5

def main():
    dirs = make_dir()
    photo_filenames = _get_filenames(dataset_dir)

    for i in range(photo_filenames.__len__()):
        print('####################################################################', i+1)
        temp = photo_filenames[i]
        base = os.path.basename(temp)
        print('>> File name :', base)
        index, probabilities = smgo_with_garbage_classifier.classify_image(photo_filenames[i])
        shutil.copy(photo_filenames[i], dirs[index])

def make_dir():
    dirname = []
    for i in range(classes.__len__()):
        dirname.append(os.path.join(dataset_dir, classes[i]))
        if not os.path.isdir(dirname[i]):
            os.mkdir(dirname[i])
    return dirname

def _get_filenames(dataset_dir):
  smgo_root = os.path.join(dataset_dir, 'images')
  photo_filenames = []
  for filename in os.listdir(smgo_root):
    path = os.path.join(smgo_root, filename)
    photo_filenames.append(path)
  return photo_filenames

if __name__ == '__main__':
    main()