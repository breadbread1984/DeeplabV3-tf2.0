#!/usr/bin/python3

from os import mkdir, listdir;
from os.path import join, exists;
import tensorflow as tf;
from models import DeeplabV3Plus;
from create_dataset import parse_function;

batch_size = 1;

def main():

  # distributed strategy
  strategy = tf.distribute.MirroredStrategy();
  # load dataset 
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  with strategy.scope():
    deeplabv3plus = DeeplabV3Plus(80 + 1);
  deeplabv3plus.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy']);
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = './checkpoints'),
    tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/ckpt', save_freq = 1000),
  ];
  deeplabv3plus.fit(trainset, epochs = 100, validation_data = testset, callbacks = callbacks);
  deeplabv3plus.save('deeplabv3plus.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
