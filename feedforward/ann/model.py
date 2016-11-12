import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.INPUT_SHAPE = [1000]
    self.HIDDEN_LAYERS = [{'shape': [200]}, {'shape': [100]}, {'shape': [100]}, {'shape': [100]}]
    self.OUTPUT_SHAPE = [100,256]
    self.LEARNING_RATE = 0.0001
    self.CLIP_GRADIENTS = 0
    self.DROPOUT = 0.8 # Keep-prob
    self.FLOAT_TYPE = tf.float32
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)

def n_dimensional_weightmul(L, W, L_shape, Lout_shape, first_dim_of_l_is_batch=True):
  """ Equivalent to matmul(W,L)
      but works for L with larger shapes than 1
      L_shape and Lout_shape are excluding the batch dimension (0)"""
  if not first_dim_of_l_is_batch:
    raise NotImplementedError
  if len(L_shape) == 1 and len(Lout_shape) == 1:
    return tf.matmul(L, W)
  # L    : ?xN1xN2xN3x...
  # Lout : ?xM1xM2xM3x...
  # W    : N1xN2x...xM1xM2x...
  # Einstein notation: letter b (denotes batch dimension)
  # Lout_blmn... = L_bijk... * Wijk...lmn...
  letters = list('ijklmnopqrst')
  l_subscripts = ''.join([letters.pop(0) for _ in range(len(L_shape))])
  lout_subscripts   = ''.join([letters.pop(0) for _ in range(len(Lout_shape))])
  einsum_string = 'b'+l_subscripts+','+l_subscripts+lout_subscripts+'->'+'b'+lout_subscripts
  return tf.einsum(einsum_string,L,W)

class Autoencoder(object):
  def __init__(self, model_params):
    self.MP = model_params

    tf.reset_default_graph()
    preset_batch_size = None
    self.variables = []
    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                              shape=[preset_batch_size] + self.MP.INPUT_SHAPE,
                                              name="input")
      if self.MP.DROPOUT is not None:
        default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
        self.dropout_placeholder = tf.placeholder_with_default(default_dropout, (), name="dropout_prob")
      self.target_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                               shape=[preset_batch_size] + self.MP.OUTPUT_SHAPE,
                                               name="output")
      self.is_sleep_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                                 shape=[preset_batch_size] + self.MP.OUTPUT_SHAPE,
                                                 name="output")
    # Encoder
    previous_layer = self.input_placeholder
    previous_layer_shape = self.MP.INPUT_SHAPE # Excludes batch dim (which should be at pos 0)
    # Flatten output
    self.shape_before_flattening = previous_layer_shape
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      with tf.name_scope('EncoderLayer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('EncoderLayer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_encoder_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_encoder_"+str(i) , dtype=self.MP.FLOAT_TYPE,
                                    shape=layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(weights)
        self.variables.append(biases)
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     weights,
                                                                     previous_layer_shape,
                                                                     layer_shape),
                                             biases),
                                      name='softplus')
        if self.MP.DROPOUT is not None:
          layer_output = tf.nn.dropout(layer_output, self.dropout_placeholder)
        # set up next loop
        previous_layer = layer_output
        previous_layer_shape = layer_shape
    # Unflatten output
    previous_layer = tf.reshape(previous_layer, shape=[-1]+self.shape_before_flattening, name="unflatten")
    previous_layer_shape = self.shape_before_flattening
    # Output (as probability of output being 1)
    self.output = tf.nn.softmax(previous_layer)
    # Loss
    with tf.name_scope('Loss') as scope:
      with tf.name_scope('ReconstructionLoss') as sub_scope:
        # Cross entropy loss of output probabilities vs. target certainties.
        reconstruction_loss = \
            -tf.reduce_sum(self.input_placeholder * tf.log(1e-10 + self.output, name="log1")
                           + (1-self.input_placeholder) * tf.log(1e-10 + (1 - self.output), name="log2"),
                           list(range(1,len(self.MP.INPUT_SHAPE)+1)))
      # Average sum of costs over batch.
      self.cost = tf.reduce_mean(reconstruction_loss + latent_loss + coercion_loss, name="cost")
    # Optimizer (ADAM)
    with tf.name_scope('Optimizer') as scope:
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE).minimize(self.cost)
      if self.MP.CLIP_GRADIENTS > 0:
        adam = tf.train.AdamOptimizer(learning_rate=self.MP.LEARNING_RATE)
        gvs = adam.compute_gradients(self.cost)
        capped_gvs = [(tf.clip_by_norm(grad, self.MP.CLIP_GRADIENTS), var) for grad, var in gvs]
        self.optimizer = adam.apply_gradients(capped_gvs)
    # Initialize session
    self.catch_nans = tf.add_check_numerics_ops()
    self.sess = tf.Session()
    tf.initialize_all_variables().run(session=self.sess)
    # Saver
    variable_names = {}
    for var in self.variables:
      variable_names[var.name] = var
    self.saver = tf.train.Saver(variable_names)

  ## Example functions for different ways to call the autoencoder graph.
  def encode(self, batch_input):
    return self.sess.run((self.z_mean, self.z_log_sigma_squared),
                         feed_dict={self.input_placeholder: batch_input})
  def decode(self, batch_z):
    return self.sess.run(self.output,
                         feed_dict={self.z_sample: batch_z})
  def encode_decode(self, batch_input):
    return self.sess.run(self.output,
                         feed_dict={self.input_placeholder: batch_input})
  def train_on_single_batch(self, batch_input, batch_latent_targets=[], cost_only=False, dropout=None):
    # feed placeholders
    dict_ = {self.input_placeholder: batch_input}
    if self.MP.DROPOUT is not None:
      dict_[self.dropout_placeholder] = self.MP.DROPOUT if dropout is None else dropout
    else:
      if dropout is not None:
        raise ValueError('This model does not implement dropout yet a value was specified')
    if len(self.MP.COERCED_LATENT_DIMS) != len(batch_latent_targets):
      print(self.MP.COERCED_LATENT_DIMS)
      print(batch_latent_targets)
      raise ValueError('latent_dim_targets are missing, but required')
    for placeholder, target in zip(self.latent_placeholders, batch_latent_targets):
      dict_[placeholder] = target
    # compute
    if cost_only:
      cost = self.sess.run(self.cost,
                           feed_dict=dict_)
    else:
      cost, _, _ = self.sess.run((self.cost, self.optimizer, self.catch_nans),
                                 feed_dict=dict_)
    return cost
  def cost_on_single_batch(self, batch_input, batch_latent_targets=[]):
    return self.train_on_single_batch(batch_input, batch_latent_targets, cost_only=True, dropout=1.0)

  def batch_encode(self, batch_input, batch_size=200, verbose=True):
    return batch_generic_func(self.encode, batch_input, batch_size, verbose)
  def batch_decode(self, batch_input, batch_size=200, verbose=True):
    return batch_generic_func(self.decode, batch_input, batch_size, verbose)
  def batch_encode_decode(self, batch_input, batch_size=100, verbose=True):
    return batch_generic_func(self.encode_decode, batch_input, batch_size, verbose)

def concat(a, b):
  if isinstance(a, list):
    return a+b
  elif isinstance(b, np.ndarray):
    return np.concatenate((a,b), axis=0)
  else:
    raise TypeError('Inputs are of unsupported type')

def batch_add(A, B):
  if isinstance(A, tuple):
    return tuple([concat(a, b) for a, b in zip(A, B)])
  else:
    return concat(A, B)

def batch_generic_func(function, batch_input, batch_size=100, verbose=False):
  a = 0
  b = batch_size
  while a < len(batch_input):
    batch_output = function(batch_input[a:b])
    try: output = batch_add(output, batch_output)
    except(NameError): output = batch_output
    a += batch_size
    b += batch_size
    if verbose: print("Example " + str(min(a,len(batch_input))) + "/" + str(len(batch_input)))
  return output
