import tensorflow as tf

def build_mlp(mlp_input, output_size, scope, n_layers, size, output_activation=None):
  print("build_mlp")
  print(output_size)
  # n_layers hidden layers with size + one output layer with output_size
  with tf.variable_scope(scope):
    x = mlp_input
    for i in range(n_layers):
      x = tf.layers.dense(x, size, activation=tf.nn.relu)
    return tf.layers.dense(x, output_size, activation=output_activation)

def build_small_cnn(cnn_input, scope):
    x=tf.layers.conv2d(cnn_input, 32, 3, 1, padding='valid', data_format='channels_last', dilation_rate=(1, 1),
                      activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=False)
    x=tf.layers.conv2d(x, 32, 3, 1, padding='valid', data_format='channels_last', dilation_rate=(1, 1),
                      activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=False)
    return tf.layers.flatten(x)

def build_cnn(cnn_input, scope):
  with tf.variable_scope(scope):
    x=tf.layers.conv2d(cnn_input, 32, 8, 4, padding='same', data_format='channels_last', dilation_rate=(1, 1),
                      activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=False)
    x=tf.layers.conv2d(x, 64, 4, 2, padding='same', data_format='channels_last', dilation_rate=(1, 1),
                      activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=False)
    x=tf.layers.conv2d(x, 64, 3, 1, padding='same', data_format='channels_last', dilation_rate=(1, 1),
                      activation=tf.nn.relu, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None,
                      bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, reuse=False)
    return tf.layers.flatten(x)
