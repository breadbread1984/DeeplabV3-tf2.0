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
  # variables for replica
  with strategy.scope():
    deeplabv3plus = DeeplabV3Plus(80 + 1);
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction = tf.keras.losses.Reduction.NONE);
    def compute_loss(labels, predictions):
      per_example_loss = loss_object(labels, predictions);
      return tf.nn.compute_average_loss(per_example_loss, global_batch_size = batch_size * strategy.num_replicas_in_sync);
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
    checkpoint = tf.train.Checkpoint(model = deeplabv3plus, optimizer = optimizer);
    train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train accuracy');
    test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test accuracy');
  # load dataset 
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  testset = tf.data.TFRecordDataset(testset_filenames).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  dist_trainset = strategy.experimental_distribute_dataset(trainset);
  dist_testset = strategy.experimental_distribute_dataset(testset);
  dist_trainset_iter = iter(dist_trainset);
  dist_testset_iter = iter(dist_testset);
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  
  # train step (per GPU)
  def train_step(inputs):
    data, labels = inputs;
    with tf.GradientTape() as tape:
      predictions = deeplabv3plus(data, training = True);
      loss = compute_loss(labels, predictions);
    gradients = tape.gradient(loss, deeplabv3plus.trainable_variables);
    if tf.math.reduce_any([tf.math.reduce_any(tf.math.logical_or(tf.math.is_nan(grad), tf.math.is_inf(grad))) for grad in gradients]) == True:
      print('detected nan in grads, skip current iterations');
      return 0;
    optimizer.apply_gradients(zip(gradients, deeplabv3plus.trainable_variables));
    train_loss.update_state(loss);
    train_accuracy.update_state(labels, predictions);
    return loss;

  def test_step(inputs):
    data, labels = inputs;
    predictions = deeplabv3plus(data, training = False);
    loss = loss_object(labels, predictions);
    test_loss.update_state(loss);
    test_accuracy.update_state(labels, predictions);

  # train
  while True:
    samples = next(dist_trainset_iter);
    # every GPU runs its replica
    per_replica_losses = strategy.run(train_step, args = (samples,));
    # reduce losses from all replica
    strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis = None);
    if tf.equal(optimizer.iterations % 100, 0):
      # save checkpoint
      checkpoint.save(join('checkpoints', 'ckpt'));
      # print test status
      for i in range(10):
        samples = next(dist_testset_iter);
        strategy.run(test_step, args = (samples,));
      # write log
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('train accuracy', train_accuracy.result(), step = optimizer.iterations);
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('test accuracy', test_accuracy.result(), step = optimizer.iterations);
      print('Step #%d Train Loss: %.6f Train Accuracy: %.6f Test Loss: %.6f Test Accuracy: %.6f' % \
          (optimizer.iterations, train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()));
      train_loss.reset_states();
      train_accuracy.reset_states();
      test_loss.reset_states();
      test_accuracy.reset_states();
      # break condition
      if train_loss.result() < 0.01: break;
  deeplabv3plus.save('deeplabv3plus.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
