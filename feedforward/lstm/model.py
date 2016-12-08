import numpy as np
import tensorflow as tf

from lstm.quantize import *

class ModelParams:
  def __init__(self):
    self.WAVE_IN_SHAPE = [1000, 1] # [timesteps, channels]
    self.WAVE_OUT_SHAPE = [100, 1]
#     self.ESTIMATOR = {'type': 'quantized', 'bins': 256, 'mu': 255} # {'type': 'gaussian'}
    self.ESTIMATOR = {'type': 'gaussian'}
    self.LEARNING_RATE = 0.001
    self.CLIP_GRADIENTS = 0
    self.DROPOUT = 0.8 # Keep-prob
    self.FLOAT_TYPE = tf.float64
    self.NUM_UNITS = 100
    self.N_LAYERS = 3
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

class LSTM(object):
  def __init__(self, model_params):
    self.MP = model_params
    self.estimator_shape = [1] if self.MP.ESTIMATOR['type'] == 'gaussian' else [self.MP.ESTIMATOR['bins']]

    tf.reset_default_graph()
    preset_batch_size = None
    self.variables = []

    # Graph input
    with tf.name_scope('Placeholders') as scope:
      self.input_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                              shape=[preset_batch_size] + self.MP.WAVE_IN_SHAPE,
                                              name="input")
      self.inputs = [self.input_placeholder[:,i] for i in range(self.MP.WAVE_IN_SHAPE[0])]
      if self.MP.DROPOUT is not None:
        default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
        self.dropout_placeholder = tf.placeholder_with_default(default_dropout, (), name="dropout_prob")
      else:
        self.dropout_placeholder = None
      self.target_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                               shape=[preset_batch_size] + self.MP.WAVE_OUT_SHAPE + self.estimator_shape,
                                               name="target")
      self.is_sleep_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                                 shape=[preset_batch_size] + self.MP.WAVE_OUT_SHAPE,
                                                 name="is_sleep")
      with tf.name_scope("StatePlaceholders") as subscope:
        self.c_state_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, shape=(preset_batch_size, self.MP.NUM_UNITS))
                                     for i in range(self.MP.N_LAYERS)]
        self.h_state_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, shape=(preset_batch_size, self.MP.NUM_UNITS))
                                     for i in range(self.MP.N_LAYERS)]
        self.initial_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state) 
                                   for c_state, h_state 
                                   in zip(self.c_state_placeholders, self.h_state_placeholders))
  
    def removable_dropout_wrapper(cell, keep_prob_placeholder):
        if keep_prob_placeholder == None:
            return cell
        else:
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob_placeholder)
    # Create stacked Cell
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
        [removable_dropout_wrapper(tf.nn.rnn_cell.LSTMCell(self.MP.NUM_UNITS, state_is_tuple=True), self.dropout_placeholder)
         for i in range(self.MP.N_LAYERS)]
                                                    , state_is_tuple=True)

    from tensorflow.python.ops import variable_scope as vs
    with tf.variable_scope("RNN") as scope:
      # Out projection layer
      previous_layer_shape = [self.MP.NUM_UNITS]
      with tf.name_scope('OutProjectionLayer') as subscope:
        layer_shape = self.MP.WAVE_OUT_SHAPE[1:] + self.estimator_shape
        flat_layer_shape = [np.prod(layer_shape)] # flatten computations
        with tf.variable_scope('OutProjectionWeights') as varscope:
          self.outprojweights = tf.get_variable("weights_out_proj", dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + flat_layer_shape)
          self.outprojbiases  = tf.get_variable("biases_out_proj" , dtype=self.MP.FLOAT_TYPE,
                                    shape=flat_layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(self.outprojweights)
        self.variables.append(self.outprojbiases)
  
      # Is sleep wave projection layer
      previous_layer_shape = [self.MP.NUM_UNITS]
      with tf.name_scope('IswProjectionLayer') as subscope:
        layer_shape = self.MP.WAVE_OUT_SHAPE[1:]
        flat_layer_shape = [np.prod(layer_shape)] # flatten computations
        with tf.variable_scope('IswProjectionWeights') as varscope:
          self.iswprojweights = tf.get_variable("weights_isw_proj", dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + flat_layer_shape)
          self.iswprojbiases  = tf.get_variable("biases_isw_proj" , dtype=self.MP.FLOAT_TYPE,
                                    shape=flat_layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(self.iswprojweights)
        self.variables.append(self.iswprojbiases)

      # Homemade seq2seq unrolling
      state = self.initial_state
      self.unrolled_outputs = []
      self.unrolled_raw_outputs = []
      self.unrolled_is_sleep = []
      for time, input_ in enumerate(self.inputs):
         if time > 0:
            scope.reuse_variables()
         (state_out, state) = stacked_lstm_cell(input_, state)
         output, _ = self.state2out(state_out)
      for i in range(self.MP.WAVE_OUT_SHAPE[0]):
         scope.reuse_variables()
         input_ = output
         (state_out, state) = stacked_lstm_cell(input_, state)
         with tf.name_scope('OutProjectionLayer') as subscope:
           with tf.variable_scope('OutProjectionWeights') as varscope:
             output, raw_output = self.state2out(state_out)
         with tf.name_scope('IswProjectionLayer') as subscope:
           with tf.variable_scope('IswProjectionWeights') as varscope:
             is_sleep = self.state2isw(state_out)
         self.unrolled_outputs.append(output)
         self.unrolled_raw_outputs.append(raw_output)
         self.unrolled_is_sleep.append(is_sleep)
      self.output = tf.pack(self.unrolled_outputs, axis=1)
      self.raw_output = tf.nn.softmax(tf.pack(self.unrolled_raw_outputs, axis=1))
      self.is_sleep = tf.pack(self.unrolled_is_sleep, axis=1)
  
    self.raw_output_shape = self.MP.WAVE_OUT_SHAPE + self.estimator_shape
    self.is_sleep_shape = self.MP.WAVE_OUT_SHAPE
    # Loss
    if self.MP.ESTIMATOR['type'] == 'gaussian': raise NotImplementedError
    with tf.name_scope('Loss') as scope:
      with tf.name_scope('ReconstructionLoss') as sub_scope:
        # Cross entropy loss of output probabilities vs. target certainties.
        if self.MP.ESTIMATOR['type'] == 'quantized':
          reconstruction_loss = \
              -tf.reduce_sum(self.target_placeholder * tf.log(1e-10 + self.raw_output, name="log1")
                             + (1-self.target_placeholder) * tf.log(1e-10 + (1 - self.raw_output), name="log2"),
                             list(range(1,len(self.raw_output_shape)+1)))
        elif self.MP.ESTIMATOR['type'] == 'gaussian':
          reconstruction_loss = \
              tf.reduce_sum(tf.square(self.target_placeholder - self.raw_output),
                             list(range(1,len(self.raw_output_shape)+1)))
      with tf.name_scope('IsSleepLoss') as sub_scope:
        # Cross entropy loss of is_sleep probabilities vs. is_sleep certainties.
        is_sleep_loss = \
            -tf.reduce_sum(self.is_sleep_placeholder * tf.log(1e-10 + self.is_sleep, name="log3")
                           + (1-self.is_sleep_placeholder) * tf.log(1e-10 + (1 - self.is_sleep), name="log4"),
                           list(range(1,len(self.is_sleep_shape)+1)))
      # Average sum of costs over batch.
      self.cost = tf.reduce_mean(reconstruction_loss + is_sleep_loss, name="cost")
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

  def estimator_output_function(self,x):
      if self.MP.ESTIMATOR['type'] == 'gaussian':
        return tf.nn.tanh(x)
      elif self.MP.ESTIMATOR['type'] == 'quantized':
        return tf.nn.softplus(x)
      else:
        raise NotImplementedError

  # Projection layer (state->output)
  def state2out(self, state):
      previous_layer = state
      previous_layer_shape = [self.MP.NUM_UNITS]
      layer_shape = self.MP.WAVE_OUT_SHAPE[1:] + self.estimator_shape
      flat_layer_shape = [np.prod(layer_shape)] # flatten computations
      layer_output = self.estimator_output_function(tf.add(n_dimensional_weightmul(previous_layer,
                                                                              self.outprojweights,
                                                                              previous_layer_shape,
                                                                              flat_layer_shape),
                                                      self.outprojbiases))
      layer_output = tf.reshape(layer_output, [-1] + layer_shape)
      return self.est2out(layer_output, layer_shape), layer_output

  # Projection layer (state->is_sleep_wave)
  def state2isw(self, state):
      previous_layer = state
      previous_layer_shape = [self.MP.NUM_UNITS]
      layer_shape = self.MP.WAVE_OUT_SHAPE[1:]
      flat_layer_shape = [np.prod(layer_shape)] # flatten computations
      layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                   self.iswprojweights,
                                                                   previous_layer_shape,
                                                                   flat_layer_shape),
                                    self.iswprojbiases))
      layer_output = tf.reshape(layer_output, [-1] + layer_shape)
      layer_output = tf.minimum(layer_output, 1)
      return layer_output

  def est2out(self, est, shape):
      if self.MP.ESTIMATOR['type'] == 'gaussian':
        return tf.reshape(est, [-1]+shape[:-1])
         # with tf.name_scope('SampleZValues') as scope:
         #     # sample = mean + sigma*epsilon
         #     epsilon = tf.random_normal(tf.shape(self.z_mean), 0, 1,
         #                                dtype=self.MP.FLOAT_TYPE)
         #     sample = tf.add(self.z_mean,
         #                     tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_squared)), epsilon))
         #     return sample
      elif self.MP.ESTIMATOR['type'] == 'quantized':
          return tf_inverse_mu_law(tf_unquantize_pick_max(est, shape), mu=self.MP.ESTIMATOR['mu'])
      else:
          raise NotImplementedError

  ## Example functions for different ways to call the autoencoder graph.
  def initialize_states(self, batch_size, feed_dict={}):
    for c_state, h_state in zip(self.c_state_placeholders, self.h_state_placeholders):
        feed_dict[c_state] = np.zeros([batch_size, self.MP.NUM_UNITS])
        feed_dict[h_state] = np.zeros([batch_size, self.MP.NUM_UNITS])
    return feed_dict

  def predict(self, batch_input):
    return self.sess.run((self.raw_output, self.is_sleep),
                         feed_dict=self.initialize_states(len(batch_input), {self.input_placeholder: batch_input}))
  def train_on_single_batch(self, batch_input, batch_target, batch_is_sleep, cost_only=False, dropout=None):
    # feed placeholders
    dict_ = {self.input_placeholder: batch_input}
    dict_[self.target_placeholder] = quantize(mu_law(batch_target),
                                              n_bins=self.MP.ESTIMATOR['bins']) if self.MP.ESTIMATOR['type'] == 'quantized' \
            else np.reshape(batch_target, [-1]+self.raw_output_shape)
    dict_[self.is_sleep_placeholder] = batch_is_sleep
    dict_ = self.initialize_states(len(batch_input), dict_)
    if self.MP.DROPOUT is not None:
      dict_[self.dropout_placeholder] = self.MP.DROPOUT if dropout is None else dropout
    else:
      if dropout is not None:
        raise ValueError('This model does not implement dropout yet a value was specified')
    # compute
    if cost_only:
      cost = self.sess.run(self.cost,
                           feed_dict=dict_)
    else:
      cost, _, _ = self.sess.run((self.cost, self.optimizer, self.catch_nans),
                                 feed_dict=dict_)
    return cost
  def cost_on_single_batch(self, batch_input, batch_target, batch_is_sleep):
    return self.train_on_single_batch(batch_input, batch_target, batch_is_sleep, cost_only=True, dropout=1.0)

  def batch_predict(self, batch_input, batch_size=200, verbose=True):
    return batch_generic_func(self.predict, batch_input, batch_size, verbose)

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
