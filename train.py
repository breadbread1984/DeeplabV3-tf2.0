#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from models import DeeplabV3Plus;
from create_dataset import parse_function;

batch_size = 1;

def main():

  # distributed strategy
  strategy = tf.distribute.MirroredStrategy();
  # load dataset 
  trainset = tf.data.TFRecordDataset(join('trainset.tfrecord')).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(join('testset.tfrecord')).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  dist_trainset = strategy.experimental_distribute_dataset(trainset);
  dist_testset = strategy.experimental_distribute_dataset(testset);
  with strategy.scope():
    deeplabv3plus = DeeplabV3Plus(3, 80 + 1);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 6000, decay_rate = 0.5));
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = "train_accuracy");
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = "test_accuracy");
    test_loss = tf.keras.metrics.Mean(name = "test_loss");
    # checkpoint
    if False == exists('checkpoints'): mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = deeplabv3plus, optimizer = optimizer);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));

    def train_step(inputs):

      images, labels = inputs;
      with tf.GradientTape() as tape:
        preds = deeplabv3plus(images);
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(labels, preds);
      gradients = tape.gradient(loss, deeplabv3plus.trainable_variables);
      optimizer.apply_gradients(zip(gradients, deeplabv3plus.trainable_variables));
      train_accuracy.update_state(labels, preds);
      return loss;
    
    def test_step(inputs):

      images, labels = inputs;
      preds = deeplabv3plus(images);
      loss = tf.keras.losses.CategoricalCrossentropy()(labels, preds);
      test_loss.update_state(loss);
      test_accuracy.update_state(labels, preds);
    
    @tf.function
    def distributed_train_step(dataset_inputs):

      per_replica_losses = strategy.experimental_run_v2(train_step, args = (dataset_inputs,));
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis = None);
    
    @tf.function
    def distributed_test_step(dataset_inputs):

      return strategy.experimental_run_v2(test_step, args = (dataset_inputs,));
    
    while True:

      for samples in dist_trainset:
        distributed_train_step(samples);
      for samples in dist_testset:
        distributed_test_step(samples);
      checkpoint.save(join('checkpoints', 'ckpt'));
      print("Step #%d Train Accuracy: %.6f Test Accuracy: %.6f Test Loss: %.6f" % (optimizer.iterations, train_accuracy.result(), test_accuracy.result(), test_loss.result()));
      train_accuracy.reset_states();
      test_accuracy.reset_states();
      test_loss.reset_states();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
