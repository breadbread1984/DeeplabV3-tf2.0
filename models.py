#!/usr/bin/python3

import tensorflow as tf;

def AtrousSpatialPyramidPooling(channel):

  inputs = tf.keras.Input((None, None, channel));
  # global pooling
  # results.shape = (batch, 1, 1, channel)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, [1,2], keepdims = True))(inputs);
  # results.shape = (batch, 1, 1, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  # pool.shape = (batch, height, width, 256)
  pool = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), method = tf.image.ResizeMethod.BILINEAR))([results, inputs]);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_1 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 6, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_6 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 12, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_12 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), dilation_rate = 18, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  dilated_18 = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height, width, 256 * 5)
  results = tf.keras.layers.Concatenate(axis = -1)([pool, dilated_1, dilated_6, dilated_12, dilated_18]);
  results = tf.keras.layers.Conv2D(256, kernel_size = (1,1), dilation_rate = 1, padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNet50(input_shape):

  inputs = tf.keras.Input(input_shape);
  resnet50 = tf.keras.applications.ResNet50(input_tensor = inputs, weights = 'imagenet', include_top = False);
  return tf.keras.Model(inputs = inputs, outputs = (resnet50.get_layer('conv4_block6_2_relu').output, resnet50.get_layer('conv2_block3_2_relu').output), name = 'resnet50');

def DeeplabV3Plus(channel = 3, nclasses = None):

  assert type(nclasses) is int;
  inputs = tf.keras.Input((None, None, channel));
  low, high = ResNet50(inputs.shape[1:])(inputs);
  # a.shape = (batch, height // 4, width // 4, 256)
  results = AtrousSpatialPyramidPooling(low.shape[-1])(low);
  a = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1] // 4, tf.shape(x[1])[2] // 4), method = tf.image.ResizeMethod.BILINEAR))([results, inputs]);
  # b.shape = (batch, height // 4, width // 4, 48)
  results = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(high);
  results = tf.keras.layers.BatchNormalization()(results);
  b = tf.keras.layers.ReLU()(results);
  # results.shape = (batch, height // 4, width // 4, 304)
  results = tf.keras.layers.Concatenate(axis = -1)([a, b]);
  # results.shape = (batch, height // 4, width // 4, 256)
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(256, kernel_size = (3,3), padding = 'same', activation = 'relu', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), method = tf.image.ResizeMethod.BILINEAR))([results, inputs]);
  results = tf.keras.layers.Conv2D(nclasses, kernel_size = (1,1), padding = 'same')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  deeplabv3 = DeeplabV3Plus(3,66);
  import numpy as np;
  results = deeplabv3(tf.constant(np.random.normal(size = (8, 224, 224, 3)), dtype = tf.float32));
  deeplabv3.save('deeplabv3.h5');
  deeplabv3 = tf.keras.models.load_model('deeplabv3.h5', compile = False);
  deeplabv3.get_layer('resnet50').save_weights('resnet50.h5');
  deeplabv3.get_layer('resnet50').load_weights('resnet50.h5');
