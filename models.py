#!/usr/bin/python3

import tensorflow as tf;

def Bottleneck(input_shape, filters, stride = 1, dilation = 1):

  # NOTE: either stride or dilation can be over 1
  inputs = tf.keras.Input(input_shape);
  residual = inputs;
  results = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters, (3, 3), padding = 'same', strides = (stride, stride), dilation_rate = (dilation, dilation), use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', use_bias = False)(results);
  results = tf.keras.layers.BatchNormalization()(results);
  if stride != 1 or inputs.shape[-1] != results.shape[-1]:
    residual = tf.keras.layers.Conv2D(filters * 4, (1, 1), padding = 'same', strides = (stride, stride), use_bias = False)(residual);
    residual = tf.keras.layers.BatchNormalization()(residual);
  results = tf.keras.layers.Add()([results, residual]);
  results = tf.keras.layers.ReLU()(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ResNetAtrous(layer_nums = [3, 4, 6, 3], dilations = [1, 2, 1]):

  strides = [2, 1, 1];
  assert layer_nums[-1] == len(dilations);
  assert len(layer_nums) == 1 + len(strides);
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Conv2D(64, (7, 7), strides = (2,2), padding = 'same', use_bias = False)(inputs);
  results = tf.keras.layers.BatchNormalization()(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = (2,2), padding = 'same')(results);
  def make_block(inputs, filters, layer_num, stride = 1, dilations = None):
    assert type(dilations) is list or dilations is None;
    results = inputs;
    for i in range(layer_num):
      results = Bottleneck(results.shape[1:], filters, stride = stride if i == 0 else 1, dilation = dilations[i] if dilations is not None else 1)(results);
    return results;
  outputs1 = make_block(results, 64, layer_nums[0]);
  outputs2 = make_block(outputs1, 128, layer_nums[1], stride = strides[0]);
  outputs3 = make_block(outputs2, 256, layer_nums[2], stride = strides[1], dilations = [1] * layer_nums[2]);
  outputs4 = make_block(outputs3, 512, layer_nums[3], stride = strides[2], dilations = dilations);
  return tf.keras.Model(inputs = inputs, outputs = (outputs1, outputs2, outputs3, outputs4));

def ResNet50Atrous():

  # NOTE: (3 + 4 + 6 + 3) * 3 + 2 = 50
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 6, 3], [1, 2, 1])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet50');

def ResNet101Atrous():

  # NOTE: (3 + 4 + 23 + 3) * 3 + 2 = 101
  inputs = tf.keras.Input((None, None, 3));
  results = ResNetAtrous([3, 4, 23, 3], [2, 2, 2])(inputs);
  return tf.keras.Model(inputs = inputs, outputs = results, name = 'resnet101');

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

def DeeplabV3Plus(nclasses = None):

  assert type(nclasses) is int;
  inputs = tf.keras.Input((None, None, 3));
  high, _, _, low = ResNet50Atrous()(inputs);
  # b.shape = (batch, height // 4, width // 4, 48)
  results = tf.keras.layers.Conv2D(48, kernel_size = (1,1), padding = 'same', kernel_initializer = tf.keras.initializers.he_normal(), use_bias = False)(high);
  results = tf.keras.layers.BatchNormalization()(results);
  b = tf.keras.layers.ReLU()(results);
  # a.shape = (batch, height // 4, width // 4, 256)
  results = AtrousSpatialPyramidPooling(low.shape[-1])(low);
  a = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], (tf.shape(x[1])[1], tf.shape(x[1])[2]), method = tf.image.ResizeMethod.BILINEAR))([results, b]);
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
  results = tf.keras.layers.Conv2D(nclasses, kernel_size = (1,1), padding = 'same', activation = tf.keras.activations.softmax, name = 'full_conv')(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  deeplabv3 = DeeplabV3Plus(66);
  import numpy as np;
  results = deeplabv3(tf.constant(np.random.normal(size = (8, 224, 224, 3)), dtype = tf.float32));
  deeplabv3.save('deeplabv3.h5');
  deeplabv3 = tf.keras.models.load_model('deeplabv3.h5', compile = False);
  # how to get the pretrained resnet50's weight
  deeplabv3.get_layer('resnet50').save_weights('resnet50.h5');
  deeplabv3.get_layer('resnet50').load_weights('resnet50.h5');
