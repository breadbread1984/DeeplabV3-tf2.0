#!/usr/bin/python3

from sys import argv;
from os.path import exists, join;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import DeeplabV3Plus;

def main(img_path):

  img = cv2.imread(img_path);
  if img is None:
    print("invalid image!");
    exit(1);
  if False == exists('models'):
    print('no pretrained model found!');
    exit(1);
  data = tf.expand_dims(img.astype('float32') / 255., axis = 0); # data.shape = (1, h, w, 3)
  if exists(join('models', 'deeplabv3plus_weights.h5')):
    deeplabv3plus = DeeplabV3Plus(1 + 80);
    deeplabv3plus.load_weights(join('models', 'deeplabv3plus_weights.h5'));
  else:
    deeplabv3plus = tf.keras.models.load_model(join('models', 'deeplabv3plus.h5'), custom_objects = {'tf': tf});
  preds = deeplabv3plus(data); # preds.shape = (1, h, w, 1 + 80)
  seg = tf.argmax(preds[0:1,...], axis = -1); # cls.shape = (1, 256, 256)
  classes, _ = tf.unique(tf.reshape(seg, (-1,))); # cls.shape = (class num)
  palette = tf.random.uniform(maxval = 256, shape = (1 + 80, 3), dtype = tf.int32); # palette.shape = (class num, 3)
  colormap = tf.cast(tf.gather_nd(palette, tf.expand_dims(seg, axis = -1)), dtype = tf.float32); # colormap.shape = (1, 255, 255, 3)
  seg_img = tf.cast(colormap, dtype = tf.uint8);
  cv2.imshow('origin', img);
  cv2.imshow('segmentation', seg_img[0].numpy());
  cv2.waitKey();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(argv) != 2:
    print("Usage: " + argv[0] + " <image>");
    exit(1);
  main(argv[1]);

