
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tensorflow as tf
import os
from slim import smgo_image_classifier
from mysite import settings

from django.db import models

from blog.models import Picture, Etc, Face, Fashion, Food, Nature, Pet

slim = tf.contrib.slim

dataset_dir = '/home/sm/PycharmProjects/test/blog/static/media'

classes = ['face', 'fashion', 'food', 'nature', 'pet', 'etc']
funcs = [Face(),Fashion(),Food(),Nature(),Pet()]
dir_path = ['face/','fashion/','food/','nature/','pet/']

MIN_PROBABIL = 0.5

class Classifier():
    def __init__(self):
        dirs = make_dir()
        photo_filenames = _get_filenames(dataset_dir)

        for i in range(photo_filenames.__len__()):
            temp = photo_filenames[i]
            base = os.path.basename(temp)
            index, probabilities = smgo_image_classifier.classify_image(photo_filenames[i])

            if (probabilities[index] < MIN_PROBABIL):
                shutil.copy(photo_filenames[i], dirs[classes.__len__() - 1])

                etc = Etc()
                etc.file = 'etc/'+os.path.basename(photo_filenames[i])
                etc.save()

            else:
                shutil.copy(photo_filenames[i], dirs[index])
                switch={
                    0:Face(),1:Fashion(),2:Food(),3:Nature(),4:Pet()
                }
                save_model = switch[index]
                save_model.file = dir_path[index] + os.path.basename(photo_filenames[i])
                save_model.save()

def make_dir():
    dirname = []
    for i in range(classes.__len__()):
        dirname.append(os.path.join(dataset_dir, classes[i]))
        if not os.path.isdir(dirname[i]):
            os.mkdir(dirname[i])
    return dirname

def _get_filenames(dataset_dir):
  smgo_root = os.path.join(dataset_dir, 'pictures')
  photo_filenames = []
  for filename in os.listdir(smgo_root):
    path = os.path.join(smgo_root, filename)
    photo_filenames.append(path)
  return photo_filenames
