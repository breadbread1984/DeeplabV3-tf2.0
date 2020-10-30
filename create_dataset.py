#!/usr/bin/python3

from os.path import join;
from pycocotools.coco import COCO;
import numpy as np;
import cv2;
import tensorflow as tf;

def create_dataset(image_dir, label_dir, trainset = True):

  anno = COCO(join(label_dir, 'instances_train2017.json' if trainset else 'instances_val2017.json'));
  writer = tf.io.TFRecordWriter('trainset.tfrecord' if trainset else 'testset.tfrecord');
  if writer is None:
    print('invalid output file!');
    exit(1);
  for image in anno.getImgIds():
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
    masks = np.stack(masks, axis = -1); # masks.shape = (h, w, 90)
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = tf.reshape(masks, (-1,))))
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  from sys import argv;
  if len(argv) != 4:
    print('Usage: %s <train image dir> <test image dir> <anno dir>' % (argv[0]));
    exit(1);
  create_dataset(argv[1], argv[3], True);
  create_dataset(argv[2], argv[3], False);
