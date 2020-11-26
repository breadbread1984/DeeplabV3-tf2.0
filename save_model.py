#!/usr/bin/python3

from os import mkdir;
from os.path import exists, join;
from shutil import rmtree;
import tensorflow as tf;
from models import DeeplabV3Plus;

def main():

  deeplabv3plus = DeeplabV3Plus(80 + 1);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  checkpoint = tf.train.Checkpoint(model = deeplabv3plus, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == exists('models'): mkdir('models');
  deeplabv3plus.save(join('models','deeplabv3plus.h5'));
  deeplabv3plus.save_weights(join('models', 'deeplabv3plus_weights.h5'));
  deeplabv3plus.get_layer('resnet50').save_weights(join('models', 'resnet50.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();

