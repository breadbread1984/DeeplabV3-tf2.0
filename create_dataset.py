#!/usr/bin/python3

from pycocoapi import coco;
import tensorflow as tf;

def create_dataset(image_dir, label_dir, trainset = True):

  anno = coco.COCO(join(label_dir, 'instances_train2017.json'));
  annotations = dict();
  # 1) collect images
  for image in labels['images']:
    img_id = image['id'];
    img_path = join(image_dir, image['file_name']);
    if exists(img_path) == False:
      print('can\'t read image ' + img_path);
      continue;
    if img_id not in annotations:
      annotations[img_id] = {'path': img_path,
                             'label': tf.zeros((0,), dtype = tf.int32), 
                             'is_crowd': tf.zeros((0,), dtype = tf.int32)};
  # 2) collect annotations
  for annotation in labels['annotations']:
    img_id = annotation['image_id'];
    if img_id not in annotations:
      print('image id %d not found' % (img_id));
      continue;
    segmentation = annotation['segmentation'][0];
    
