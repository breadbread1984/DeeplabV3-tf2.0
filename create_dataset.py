#!/usr/bin/python3

from os.path import join;
from math import ceil;
from multiprocessing import Process, Lock;
from pycocotools.coco import COCO;
import numpy as np;
import cv2;
import tensorflow as tf;

PROCESS_NUM = 64;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'image': tf.io.FixedLenFeature((), dtype = tf.string, default_value = ''),
      'shape': tf.io.FixedLenFeature((3,), dtype = tf.int64),
      'label': tf.io.VarLenFeature(dtype = tf.float32)
    }
  );
  shape = tf.cast(feature['shape'], dtype = tf.int32);
  data = tf.io.decode_jpeg(feature['image']);
  data = tf.reshape(data, shape);
  data = tf.cast(data, dtype = tf.float32);
  label = tf.sparse.to_dense(feature['label'], default_value = 0);
  label = tf.reshape(label, (shape[0], shape[1], -1)); # label.shape = (h, w, 80)
  tf.debugging.Assert(tf.math.equal(label.shape[-1], 80), label.shape);
  foreground_mask = tf.math.reduce_max(label, axis = -1);
  background_mask = tf.where(tf.math.equal(foreground_mask, 0), tf.ones_like(foreground_mask), tf.zeros_like(foreground_mask)); # background_mask.shape = (h, w)
  label = tf.concat([tf.expand_dims(background_mask, axis = -1), label], axis = -1); # label.shape = (h, w, 81)
  return data, label;

def create_dataset(image_dir, label_dir, trainset = True):

  anno = COCO(join(label_dir, 'instances_train2017.json' if trainset else 'instances_val2017.json'));
  writer = tf.io.TFRecordWriter('trainset.tfrecord' if trainset else 'testset.tfrecord');
  if writer is None:
    print('invalid output file!');
    exit(1);
  imgs_for_each = ceil(len(anno.getImgIds()) / PROCESS_NUM);
  handlers = list();
  lock = Lock();
  for i in range(PROCESS_NUM):
    handlers.append(Process(target = worker, args = (anno.getImgIds()[i * imgs_for_each:(i+1) * imgs_for_each] if i != PROCESS_NUM - 1 else anno.getImgIds()[i * imgs_for_each:], lock)));
    handlers[-1].start();
  for handler in handlers:
    handler.join();
  writer.close();

def worker(image_ids, lock)
  for image in img_ids:
    img_info = anno.loadImgs([image])[0];
    img = cv2.imread(join(image_dir, img_info['file_name']));
    if img is None:
      print('can\'t open image %s' % (join(image_dir, img_info['file_name'])));
      continue;
    masks = list();
    for category in anno.getCatIds():
      annIds = anno.getAnnIds(imgIds = image, catIds = category);
      mask = np.zeros((img_info['height'], img_info['width']));
      anns = anno.loadAnns(annIds);
      for ann in anns:
        # for every instance of category in current image
        instance_mask = anno.annToMask(ann);
        mask = np.maximum(mask, instance_mask);
      masks.append(mask);
    masks = np.stack(masks, axis = -1); # masks.shape = (h, w, 80)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
        'label': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(masks, (-1,))))
      }
    ));
    lock.acquire();
    writer.write(trainsample.SerializeToString());
    lock.release();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  from sys import argv;
  if len(argv) != 4:
    print('Usage: %s <train image dir> <test image dir> <anno dir>' % (argv[0]));
    exit(1);
  create_dataset(argv[1], argv[3], True);
  create_dataset(argv[2], argv[3], False);
