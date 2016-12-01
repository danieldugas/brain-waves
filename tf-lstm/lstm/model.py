import numpy as np
import tensorflow as tf

class ModelParams:
  def __init__(self):
    self.WAVE_IN_SHAPE = [1000, 1] # [timesteps, channels]
    self.WAVE_OUT_SHAPE = [10, 1]
    self.ESTIMATOR = {'type': 'quantized', 'bins': 256} # {'type': 'variational'}
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

class LSTM(object):
  def __init__(self, model_params):
    self.MP = model_params
    self.estimator_shape = [2] if self.MP.ESTIMATOR['type'] == 'variational' else [self.MP.ESTIMATOR['bins']]

    tf.reset_default_graph()
    preset_batch_size = None
    self.variables = []
  
  
  
  # Initialize session and write graph for visualization.
  sess = tf.Session()
  tf.initialize_all_variables().run(session=sess)

    # Graph input
    with tf.name_scope('Placeholders') as scope:
      # Placeholders
      if self.MP.DROPOUT is not None:
        default_dropout = tf.constant(1, dtype=self.MP.FLOAT_TYPE)
        self.dropout_placeholder = tf.placeholder_with_default(default_dropout, (), name="dropout_prob")
      else:
        self.dropout_placeholder = None
      with tf.name_scope("InputPlaceholders") as subscope:
        self.input_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, 
                                                  shape=[preset_batch_size] +
                                                  self.MP.WAVE_IN_SHAPE,
                                                  name="input")
                                   for i in range(self.MP.OUTSET_CUTOFF)]
      with tf.name_scope("StatePlaceholders") as subscope:
        self.c_state_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, shape=(preset_batch_size, self.MP.NUM_UNITS), name="c_state"+str(i))
                                     for i in range(self.MP.N_LAYERS)]
        self.h_state_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, shape=(preset_batch_size, self.MP.NUM_UNITS), name="h_state"+str(i)) 
                                     for i in range(self.MP.N_LAYERS)]
        self.initial_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state) 
                                   for c_state, h_state 
                                   in zip(self.c_state_placeholders, self.h_state_placeholders))
      with tf.name_scope("TargetPlaceholder") as subscope:
        self.target_placeholders = [tf.placeholder(self.MP.FLOAT_TYPE, shape=(preset_batch_size, self.MP.OUTPUT_SIZE), name="target"+str(i))
                                    for i in
                                    range(self.MP.BPTT_LENGTH-self.MP.WAVE_IN_SHAPE[0])]
        self.is_sleep_placeholder = tf.placeholder(self.MP.FLOAT_TYPE,
                                                   shape=[preset_batch_size] + self.MP.WAVE_OUT_SHAPE,
                                                   name="output")

    def removable_dropout_wrapper(cell, keep_prob_placeholder):
        if keep_prob_placeholder == None:
            return cell
        else 
            return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob_placeholder)



    # Cells
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
        [removable_dropout_wrapper(tf.nn.rnn_cell.LSTMCell(self.MP.NUM_UNITS, state_is_tuple=True), self.dropout_placeholder)
         for i in range(self.MP.N_LAYERS)]
                                                    , state_is_tuple=True)
    # Homemade seq2seq unrolling of the cells
    from tensorflow.python.ops import variable_scope as vs
    state = self.initial_state
    self.unrolled_outputs = []
    with tf.variable_scope("RNN") as scope:
        for time, input_ in enumerate(input_placeholders):
           if time > 0:
              scope.reuse_variables()
           (state_out, state) = stacked_lstm_cell(input_, state)
           output = self.state2out(state_out)
        for i in range(MP.WAVE_OUT_SHAPE[0]):
           scope.reuse_variables()
           input_ = self.out2in(output)
           (state_out, state) = stacked_lstm_cell(input_, state)
           output = self.state2out(state_out)
           self.unrolled_outputs.append(output)


    previous_layer = self.input_placeholder
    previous_layer_shape = self.MP.WAVE_IN_SHAPE # Excludes batch dim (which should be at pos 0)
    # Flatten output
    self.shape_before_flattening = previous_layer_shape
    previous_layer_shape = [np.prod(previous_layer_shape)]
    previous_layer = tf.reshape(previous_layer, shape=[-1]+previous_layer_shape, name="flatten")
    # Fully connected Layers
    for i, LAYER in enumerate(self.MP.HIDDEN_LAYERS):
      with tf.name_scope('Layer'+str(i)) as scope:
        layer_shape = LAYER['shape']
        with tf.variable_scope('Layer'+str(i)+'Weights') as varscope:
          weights = tf.get_variable("weights_layer_"+str(i), dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          biases  = tf.get_variable("biases_layer_"+str(i) , dtype=self.MP.FLOAT_TYPE,
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
    # Output (as probability of output being 1)
    with tf.name_scope('OutputLayer') as scope:
      layer_shape = self.MP.WAVE_OUT_SHAPE + [self.MP.QUANTIZATION]
      layer_shape = [np.prod(layer_shape)] # Compute as a flat layer
      with tf.variable_scope('OutputLayerWeights') as varscope:
        weights = tf.get_variable("weights_output", dtype=self.MP.FLOAT_TYPE,
                                  shape=previous_layer_shape + layer_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases_output" , dtype=self.MP.FLOAT_TYPE,
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
      # Unflatten output
      output_shape = self.MP.WAVE_OUT_SHAPE + [self.MP.QUANTIZATION]
      unflattened = tf.reshape(layer_output, shape=[-1]+output_shape, name="unflatten")
      self.output = tf.nn.softmax(unflattened)
    # Sleep prediction
    with tf.name_scope('IsSleepLayer') as scope:
      layer_shape = self.MP.WAVE_OUT_SHAPE
      layer_shape = [np.prod(layer_shape)] # Compute as a flat layer
      with tf.variable_scope('IsSleepLayerWeights') as varscope:
        weights = tf.get_variable("weights_is_sleep", dtype=self.MP.FLOAT_TYPE,
                                  shape=previous_layer_shape + layer_shape,
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases  = tf.get_variable("biases_is_sleep" , dtype=self.MP.FLOAT_TYPE,
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
      # Unflatten output
      is_sleep_shape = self.MP.WAVE_OUT_SHAPE
      self.is_sleep = tf.minimum(tf.reshape(layer_output, shape=[-1]+is_sleep_shape, name="unflatten"), 1)
    # Loss
    with tf.name_scope('Loss') as scope:
      with tf.name_scope('ReconstructionLoss') as sub_scope:
        # Cross entropy loss of output probabilities vs. target certainties.
        reconstruction_loss = \
            -tf.reduce_sum(self.target_placeholder * tf.log(1e-10 + self.output, name="log1")
                           + (1-self.target_placeholder) * tf.log(1e-10 + (1 - self.output), name="log2"),
                           list(range(1,len(output_shape)+1)))
      with tf.name_scope('IsSleepLoss') as sub_scope:
        # Cross entropy loss of is_sleep probabilities vs. is_sleep certainties.
        is_sleep_loss = \
            -tf.reduce_sum(self.is_sleep_placeholder * tf.log(1e-10 + self.is_sleep, name="log3")
                           + (1-self.is_sleep_placeholder) * tf.log(1e-10 + (1 - self.is_sleep), name="log4"),
                           list(range(1,len(is_sleep_shape)+1)))
      # Average sum of costs over batch.
      self.cost = tf.reduce_mean(reconstruction_loss + is_sleep_loss, name="cost")
  # Loss
  with tf.variable_scope("CostFunction") as scope:
    loss = [tf.square(target_placeholder - output, name="loss"+str(i)) 
            for i, (target_placeholder, output) in enumerate(zip(target_placeholders, outputs[MP.OUTSET_CUTOFF:]))]
    loss = tf.add_n(loss, name="summed_seq_loss") # add together losses for each sequence step
    electrode_loss_weights = tf.constant([1.0]+[0.01 for _ in range(MP.OUTPUT_SIZE-1)], name="loss_weights")
    loss = tf.mul(loss, electrode_loss_weights, name="weighted_loss") # weigh loss for each electrode
    if MP.OUTPUT_SIZE > 1:
        loss = tf.reduce_sum(loss, 1) # add together loss for all outputs
    cost = tf.reduce_mean(loss, name="cost")   # average over batch
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

  # Projection layer (state->output)
  def state2out(self, state, with_estimator=True):
      previous_layer = state
      previous_layer_shape = [self.MP.NUM_UNITS]
      with tf.name_scope('OutProjectionLayer') as scope:
        layer_shape = self.MP.WAVE_OUT_SHAPE[1:] + self.estimator_shape if with_estimator else self.MP.WAVE_OUT_SHAPE[1:]
        flat_layer_shape = np.prod(layer_shape) # flatten computations
        with tf.variable_scope('OutProjectionWeights') as varscope:
          varscope.reuse_variables()
          outprojweights = tf.get_variable("weights_out_proj", dtype=self.MP.FLOAT_TYPE,
                                    shape=previous_layer_shape + flat_layer_shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
          outprojbiases  = tf.get_variable("biases_out_proj" , dtype=self.MP.FLOAT_TYPE,
                                    shape=flat_layer_shape,
                                    initializer=tf.constant_initializer(0))
        self.variables.append(outprojweights)
        self.variables.append(outprojbiases)
        layer_output = tf.nn.softplus(tf.add(n_dimensional_weightmul(previous_layer,
                                                                     outprojweights,
                                                                     previous_layer_shape,
                                                                     flat_layer_shape),
                                             outprojbiases),
                                      name='softplus')
        layer_output = tf.reshape(layer_output, [-1] + layer_shape)
      return layer_output

  # Projection layer (state->is_sleep_wave)
  def state2isw(self, state):
      return self.state2out(state, with_estimator=False)

  def out2in(self, out):
      if self.MP.ESTIMATOR['type'] == 'variational':
          with tf.name_scope('SampleZValues') as scope:
              # sample = mean + sigma*epsilon
              epsilon = tf.random_normal(tf.shape(self.z_mean), 0, 1,
                                         dtype=self.MP.FLOAT_TYPE, name='randomnormal')
              sample = tf.add(self.z_mean,
                              tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_squared)), epsilon),
                              name='z_sample')
              return sample
      elif self.MP.ESTIMATOR['type'] == 'quantized':
          return inverse_mu_law(unquantize(pick_max(out)))
      else:
          raise NotImplementedError

  ## Example functions for different ways to call the autoencoder graph.
  def predict(self, batch_input):
    return self.sess.run((self.output, self.is_sleep),
                         feed_dict={self.input_placeholder: batch_input})
  def train_on_single_batch(self, batch_input, batch_target, batch_is_sleep, cost_only=False, dropout=None):
    # feed placeholders
    from ann.quantize import quantize, mu_law
    dict_ = {self.input_placeholder: batch_input}
    dict_[self.target_placeholder] = quantize(mu_law(batch_target), n_bins=self.MP.QUANTIZATION)
    dict_[self.is_sleep_placeholder] = batch_is_sleep
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
