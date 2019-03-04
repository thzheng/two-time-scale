import tensorflow as tf

def build_mlp(mlp_input, output_size, scope, n_layers, size, output_activation=None):
  # n_layers hidden layers with size + one output layer with output_size
  with tf.variable_scope(scope):
    x = mlp_input
    for i in range(n_layers):
      x = tf.layers.dense(x, size, activation=tf.nn.relu)
    return tf.layers.dense(x, output_size, activation=output_activation)
