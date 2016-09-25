
# coding: utf-8

# In[ ]:

from __future__ import print_function

import tensorflow as tf
import numpy as np


# ## Parameters

# In[ ]:

PLOTTING_SUPPORT = True
SET_EULER_PARAMETERS = False

# Handle arguments (When executed as .py script)
import sys
argv = sys.argv[:]
if len(argv) > 1:
  script_path = argv.pop(0)
  if "--euler" in argv:
    SET_EULER_PARAMETERS = True
    PLOTTING_SUPPORT = False
    print("Parameters set for execution on euler cluster")
    argv.remove("--euler")


# In[ ]:

DATA_FOLDER = "/home/daniel/Downloads/Data_200Hz/"
DATA_FILENAME="077_COSession1.set"
DATA2_FILENAME="077_COSession2.set"
ELECTRODES_OF_INTEREST = ['E36','E22','E9','E33','E24','E11','E124','E122','E45','E104',
                          'E108','E58','E52','E62','E92','E96','E70','E83','E75']
BATCH_SIZE = 20
BATCH_LIMIT_PER_STEP = 10
TRAINING_DATA_LENGTH = "max"
VAL_DATA_LENGTH = 1200000
TEST_DATA_LENGTH = 2000
SAMPLING = 1

MAX_STEPS = 1000

VAL_EVERY_N_STEPS = 1
VAL_STEP_TOLERANCE = 100

class ModelParams:
  def __init__(self):
    self.BPTT_LENGTH = 100
    self.OUTSET_CUTOFF = 50
    self.NUM_UNITS = 512
    self.N_LAYERS = 3
    self.INPUT_SIZE = 19
    self.OUTPUT_SIZE = 19
    self.LEARNING_RATE = 0.001
    self.CLIP_GRADIENTS = 1.0
    self.SCALE_OUTPUT = 100.0
    self.DROPOUT = 1.0
  def __str__(self):
    return str(self.__dict__)
  def __eq__(self, other): 
    return self.__dict__ == other.__dict__
  def __ne__(self, other):
    return not self.__eq__(other)
MP = ModelParams()

SAVE_DIR = "/home/daniel/Desktop/tf-lstm-model/"
SAVE_FILE = "model.ckpt"
TENSORBOARD_DIR = "/home/daniel/tensorboard"
#DATA_FOLDER = "/home/daniel/Downloads/Raw-Waves/"
#DATA_FILENAME="001_Session1_FilterTrigCh_RawCh.mat"
#DATA2_FILENAME="001_Session2_FilterTrigCh_RawCh.mat"
#DATA3_FILENAME="034_Session1_FilterTrigCh_RawCh.mat"

assert MP.INPUT_SIZE == len(ELECTRODES_OF_INTEREST)
assert MP.INPUT_SIZE == MP.OUTPUT_SIZE


# In[ ]:

if SET_EULER_PARAMETERS:
    DATA_FOLDER = "/cluster/home/dugasd/Data_200Hz/"
    SAVE_DIR = "/cluster/home/dugasd/tf-lstm-model/"
    TENSORBOARD_DIR = None
    
    BATCH_SIZE = 20
    BATCH_LIMIT_PER_STEP = 10000
    MAX_STEPS = 1000000
    VAL_STEP_TOLERANCE = 100


# In[ ]:

if PLOTTING_SUPPORT:
  import matplotlib.pyplot as plt
  get_ipython().magic('matplotlib inline')
  from cycler import cycler
  if SAMPLING > 0:
      plotting_function = plt.step
  else:
      plotting_function = plt.plot


# In[ ]:

SAVE_PATH = SAVE_DIR+SAVE_FILE


# ## Datasets

# In[ ]:

if False:
  raw_wave = []
  raw_wave2 = []
  raw_wave3 = []

  import scipy.io
  mat = scipy.io.loadmat(DATA_FOLDER+DATA_FILENAME)
  raw_wave = mat.get('data')[0]
  raw_wave = raw_wave[::SAMPLING]
  raw_wave = raw_wave[0:]

  if DATA2_FILENAME is not None:
      mat = scipy.io.loadmat(DATA_FOLDER+DATA2_FILENAME)
      raw_wave2 = mat.get('data')[0]
      raw_wave2 = raw_wave2[::SAMPLING]
      raw_wave2 = raw_wave2[0:]
  if DATA3_FILENAME is not None:
      mat = scipy.io.loadmat(DATA_FOLDER+DATA3_FILENAME)
      raw_wave3 = mat.get('data')[0]
      raw_wave3 = raw_wave3[::SAMPLING]
      raw_wave3 = raw_wave3[0:]
    
  # Save some memory
  del mat


# In[ ]:

if False:
  raw_wave = []
  raw_wave2 = []
  raw_wave3 = []
 
  import mne
  raw_eeglab = mne.io.read_raw_eeglab(DATA_FOLDER+DATA_FILENAME)
  electrode_names = raw_eeglab.ch_names
  EOI_indices = [electrode_names.index(name) for name in ELECTRODES_OF_INTEREST]
  raw_wave = np.array([raw_eeglab[e_index][0][0] for e_index in EOI_indices])
  raw_wave = list(raw_wave.T)
  raw_wave = raw_wave[::SAMPLING]*1000000

  raw_eeglab = mne.io.read_raw_eeglab(DATA_FOLDER+DATA2_FILENAME)
  electrode_names = raw_eeglab.ch_names
  EOI_indices = [electrode_names.index(name) for name in ELECTRODES_OF_INTEREST]
  raw_wave2 = np.array([raw_eeglab[e_index][0][0] for e_index in EOI_indices])
  raw_wave2 = list(raw_wave2.T)
  raw_wave2 = raw_wave2[::SAMPLING]*1000000

  del raw_eeglab


# In[ ]:

if False:
  ARTEFACTS_FILENAME = "077_Session1_artefact.mat"
  import scipy.io
  mat = scipy.io.loadmat(DATA_FOLDER+ARTEFACTS_FILENAME)
  artefacts = mat.get('artndxn')
  artefacts = np.sum(artefacts, axis=0)

  ARTEFACTS2_FILENAME = "077_Session2_artefact.mat"
  mat = scipy.io.loadmat(DATA_FOLDER+ARTEFACTS2_FILENAME)
  artefacts2 = mat.get('artndxn')
  artefacts2 = np.sum(artefacts2, axis=0)
  # Save some memory
  del mat

  artefacts = np.pad(artefacts[:,None], ((0,0),(0,20*200)), 'edge').flatten()
  print(raw_wave.shape)
  print(artefacts.shape)

  artefacts2 = np.pad(artefacts2[:,None], ((0,0),(0,20*200)), 'edge').flatten()
  print(raw_wave2.shape)
  print(artefacts2.shape)

  plt.figure(figsize=(100,10))
  plotting_function(np.arange(len(artefacts)), artefacts)
  plotting_function(range(len(raw_wave)), raw_wave[:,:3])
  plt.ylim([-300,300])
  plt.show()
    
  raw_wave = raw_wave[np.where(artefacts == artefacts.max())]
  del artefacts
  raw_wave2 = raw_wave2[np.where(artefacts2 == artefacts2.max())]
  del artefacts2


# In[ ]:

if False:
  np.save(DATA_FOLDER+"raw_wave", raw_wave)
  np.save(DATA_FOLDER+"raw_wave2", raw_wave2)


# In[ ]:

if True:
  raw_wave  = np.load(DATA_FOLDER+"raw_wave.npy")
  raw_wave2 = np.load(DATA_FOLDER+"raw_wave2.npy")
  raw_wave  = raw_wave[::SAMPLING]
  raw_wave2 = raw_wave2[::SAMPLING]
  raw_wave3 = []


# In[ ]:

if TRAINING_DATA_LENGTH == "max":
    TRAINING_DATA_LENGTH = len(raw_wave)
if VAL_DATA_LENGTH == "max":
    assert len(raw_wave2) != 0
    VAL_DATA_LENGTH = len(raw_wave2) - TEST_DATA_LENGTH
assert TRAINING_DATA_LENGTH + VAL_DATA_LENGTH + TEST_DATA_LENGTH <= len(raw_wave) + len(raw_wave2) + len(raw_wave3)


# In[ ]:

# Assign data to datasets.
training_data = raw_wave[:TRAINING_DATA_LENGTH]
if len(raw_wave2) is 0:
  val_data = raw_wave[TRAINING_DATA_LENGTH:][:VAL_DATA_LENGTH]
  test_data = raw_wave[TRAINING_DATA_LENGTH:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
else:
  val_data = raw_wave2[:VAL_DATA_LENGTH]
  if len(raw_wave3) is 0:
    test_data = raw_wave2[VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
  else:
    test_data = raw_wave3[:TEST_DATA_LENGTH]


if PLOTTING_SUPPORT:
  plt.figure(figsize=(500,10))
  plotting_function(range(TRAINING_DATA_LENGTH),training_data[:,0],label="training")
  plotting_function(range(TRAINING_DATA_LENGTH,TRAINING_DATA_LENGTH+VAL_DATA_LENGTH),val_data[:,0],label="validation")
  plotting_function(range(TRAINING_DATA_LENGTH+VAL_DATA_LENGTH,
                 TRAINING_DATA_LENGTH+VAL_DATA_LENGTH+TEST_DATA_LENGTH),test_data[:,0],label="test")
  plt.ylim([-1000,1000])
  plt.figure(figsize=(500,10))
  plotting_function(range(TRAINING_DATA_LENGTH),training_data[:,0],label="training")
  plt.xlim([0, 100000])
  plt.ylim([-100,100])
  #plt.legend()
print(len(raw_wave)-TRAINING_DATA_LENGTH)
print(len(raw_wave2)-VAL_DATA_LENGTH-TEST_DATA_LENGTH)


# ## Model Initialization

# In[ ]:

import os
if os.path.exists(SAVE_PATH):
  with open(SAVE_DIR+"model_params.pckl", 'rb') as file:
    import pickle
    mp_loaded = pickle.load(file)
  if MP != mp_loaded:
    print(MP)
    print(mp_loaded)
  assert MP == mp_loaded


# In[ ]:

## Create Graph
tf.reset_default_graph()

preset_batch_size = None
# Placeholders
with tf.name_scope("input_placeholders") as scope:
  input_placeholders = [tf.placeholder(tf.float32, shape=(preset_batch_size, MP.INPUT_SIZE), name="input"+str(i))
                        for i in range(MP.OUTSET_CUTOFF)]
dropout_placeholder = tf.placeholder(tf.float32, name="dropout_prob")
c_state_placeholders = [tf.placeholder(tf.float32, shape=(preset_batch_size, MP.NUM_UNITS), name="c_state"+str(i)) for i in range(MP.N_LAYERS)]
h_state_placeholders = [tf.placeholder(tf.float32, shape=(preset_batch_size, MP.NUM_UNITS), name="h_state"+str(i)) for i in range(MP.N_LAYERS)]
initial_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state) for c_state, h_state in zip(c_state_placeholders, h_state_placeholders))
with tf.name_scope("target_placeholders") as scope:
  target_placeholders = [tf.placeholder(tf.float32, shape=(preset_batch_size, MP.OUTPUT_SIZE), name="target"+str(i))
                        for i in range(MP.BPTT_LENGTH-MP.OUTSET_CUTOFF)]

# Cells
stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(MP.NUM_UNITS, state_is_tuple=True),
                                   output_keep_prob=dropout_placeholder) for i in range(MP.N_LAYERS)],
                                                state_is_tuple=True)
from tensorflow.python.ops import variable_scope as vs
state = initial_state
unrolled_outputs = []
with tf.variable_scope("RNN") as scope:
    for time, input_ in enumerate(input_placeholders):
       if time > 0:
          scope.reuse_variables()
       (output, state) = stacked_lstm_cell(input_, state)
       unrolled_outputs.append(output)
    for i in range(MP.BPTT_LENGTH-MP.OUTSET_CUTOFF):
       scope.reuse_variables()
       (output, state) = stacked_lstm_cell(output[:, 0:MP.OUTPUT_SIZE], state)
       unrolled_outputs.append(output)
        
outputs = [tf.mul(cell_output[:, 0:MP.OUTPUT_SIZE], tf.constant(MP.SCALE_OUTPUT)) for cell_output in unrolled_outputs]

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

# ADAM optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=MP.LEARNING_RATE).minimize(cost)
if MP.CLIP_GRADIENTS > 0:
  adam = tf.train.AdamOptimizer(learning_rate=MP.LEARNING_RATE)
  gvs = adam.compute_gradients(cost)
#  capped_gvs = [(tf.clip_by_value(grad, -MP.CLIP_GRADIENTS, MP.CLIP_GRADIENTS), var) for grad, var in gvs]
  capped_gvs = [(tf.clip_by_norm(grad, MP.CLIP_GRADIENTS), var) for grad, var in gvs]
  optimizer = adam.apply_gradients(capped_gvs)

# Initialize session and write graph for visualization.
sess = tf.Session()
tf.initialize_all_variables().run(session=sess)

print("Session created.")


# In[ ]:

if TENSORBOARD_DIR != None:
  summary_writer = tf.train.SummaryWriter(TENSORBOARD_DIR, sess.graph)
  print("Tensorboard graph saved.")


# In[ ]:

saver = tf.train.Saver()

import pickle
mp_filename = "model_params.pckl"


# In[ ]:

# Restore model weights from previously saved model
import os
if os.path.exists(SAVE_PATH):
  saver.restore(sess, SAVE_PATH)
  print("Model restored from file: %s" % SAVE_PATH)
else:
  print("No model found.")


# ### Training

# In[ ]:

from batchmaker import StatefulBatchmaker

total_step_cost = None
step_cost_log = []
total_val_cost = 0
val_cost_log = []
val_steps_since_last_improvement = 0
make_new_training_batches_and_states = True

# single step
for step in range(MAX_STEPS):
  # Validation
  val_batchmaker = StatefulBatchmaker(val_data, MP.BPTT_LENGTH, MP.OUTSET_CUTOFF, BATCH_SIZE, MP.OUTPUT_SIZE)
  prev_batch_c_states = [np.zeros((BATCH_SIZE, MP.NUM_UNITS)) for i in range(MP.N_LAYERS)]
  prev_batch_h_states = [np.zeros((BATCH_SIZE, MP.NUM_UNITS)) for i in range(MP.N_LAYERS)]
  if np.mod(step, VAL_EVERY_N_STEPS) == 0:
    total_val_cost = 0
    while True:
      if BATCH_LIMIT_PER_STEP > 0:
        if val_batchmaker.n_batches_consumed() > BATCH_LIMIT_PER_STEP:
          break
      if val_batchmaker.is_depleted():
        break
      else:
        batch_input_values, batch_target_values = val_batchmaker.next_batch()
    
        # Assign a value to each placeholder.
        feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
        feed_dictionary[dropout_placeholder] = 1.0
        for target_placeholder, target_value in zip(target_placeholders, batch_target_values):
            feed_dictionary[target_placeholder] = target_value
        for c_state_placeholder, prev_batch_c_state in zip(c_state_placeholders, prev_batch_c_states):
            feed_dictionary[c_state_placeholder] = prev_batch_c_state
        for h_state_placeholder, prev_batch_h_state in zip(h_state_placeholders, prev_batch_h_states):
            feed_dictionary[h_state_placeholder] = prev_batch_h_state

       # Validate.
        cost_value, state_value = sess.run((cost, state), feed_dict=feed_dictionary)
        total_val_cost += cost_value
        prev_batch_c_states = [state_value[i].c for i in range(len(state_value))]
        prev_batch_h_states = [state_value[i].h for i in range(len(state_value))]
    print("Validation cost: ", end='')
    print(total_val_cost, end='')
    print("  (Training cost: ", end='')
    print(total_step_cost, end='')
    print(")")
    val_cost_log.append(total_val_cost)
    
    # Training Monitor
    if len(val_cost_log) > 1:
        # Save cost log.
        import os
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            print("Created directory: %s" % SAVE_DIR)
        np.savetxt(SAVE_DIR+"val_cost_log.txt", val_cost_log)
        # Save if cost has improved. Otherwise increment counter.
        if val_cost_log[-1] <  min(val_cost_log[:-1]):
            val_steps_since_last_improvement = 0
            # save model to disk
            print("Saving ... ", end='')
            save_path = saver.save(sess, SAVE_PATH)
            with open(SAVE_DIR+mp_filename, 'wb') as file:
              pickle.dump(MP, file, protocol=2)
            print("Model saved in file: %s" % save_path)      
        else:
            val_steps_since_last_improvement += 1         
    # Stop training if val_cost hasn't improved in VAL_STEP_TOLERANCE steps
    if val_steps_since_last_improvement > VAL_STEP_TOLERANCE:
        print("Training stopped by validation monitor.")
        break
            
  # Train on batches
  total_step_cost = 0
  step_batches_counter = 0
  if make_new_training_batches_and_states:
    training_batchmaker = StatefulBatchmaker(training_data, MP.BPTT_LENGTH, MP.OUTSET_CUTOFF, BATCH_SIZE, MP.OUTPUT_SIZE)
    prev_batch_c_states = [np.zeros((BATCH_SIZE, MP.NUM_UNITS)) for i in range(MP.N_LAYERS)]
    prev_batch_h_states = [np.zeros((BATCH_SIZE, MP.NUM_UNITS)) for i in range(MP.N_LAYERS)]
    make_new_training_batches_and_states = False
  while True:
    if BATCH_LIMIT_PER_STEP > 0:
      step_batches_counter += 1
      if step_batches_counter > BATCH_LIMIT_PER_STEP:
        break
    if training_batchmaker.is_depleted():
      make_new_training_batches_and_states = True
      break
    else:
      batch_input_values, batch_target_values = training_batchmaker.next_batch()
      
      # Assign a value to each placeholder.
      feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
      feed_dictionary[dropout_placeholder] = MP.DROPOUT
      for target_placeholder, target_value in zip(target_placeholders, batch_target_values):
        feed_dictionary[target_placeholder] = target_value
      for c_state_placeholder, prev_batch_c_state in zip(c_state_placeholders, prev_batch_c_states):
        feed_dictionary[c_state_placeholder] = prev_batch_c_state
      for h_state_placeholder, prev_batch_h_state in zip(h_state_placeholders, prev_batch_h_states):
        feed_dictionary[h_state_placeholder] = prev_batch_h_state
  
      # Train over 1 batch.
      opt_value, last_output_value, cost_value, state_value = sess.run((optimizer, outputs[-1], cost, state),
                                                              feed_dict=feed_dictionary)
      total_step_cost += cost_value
      prev_batch_c_states = [state_value[i].c for i in range(len(state_value))]
      prev_batch_h_states = [state_value[i].h for i in range(len(state_value))]
      assert not np.isnan(last_output_value).any()
  step_cost_log.append(total_step_cost)


print("Training ended.")


# In[ ]:

if PLOTTING_SUPPORT:
  plt.figure(figsize=(100,10))
  plotting_function(range(len(step_cost_log)), step_cost_log, label="step_cost_log")
  plotting_function(range(len(val_cost_log)), val_cost_log, label="val_cost_log")
  plt.legend()


# In[ ]:

# Restore model weights from previously saved model
import os
if os.path.exists(SAVE_PATH):
  saver.restore(sess, SAVE_PATH)
  print("Model restored from file: %s" % SAVE_PATH)
else:
  print("No model found.")


# ### Testing

# In[ ]:

offset = 0


# In[ ]:

REALIGN_OUTPUT = True

from batchmaker import StatefulBatchmaker
test_batchmaker = StatefulBatchmaker(test_data, MP.BPTT_LENGTH, MP.OUTSET_CUTOFF, 1, MP.OUTPUT_SIZE, True)


testing_cost = 0
test_outputs = []
prev_batch_c_states = [np.zeros((1, MP.NUM_UNITS)) for i in range(len(state_value))]
prev_batch_h_states = [np.zeros((1, MP.NUM_UNITS)) for i in range(len(state_value))]
while True:
  if test_batchmaker.is_depleted():
    break
  else:
    batch_input_values, batch_target_values = test_batchmaker.next_batch()
    
    # Assign a value to each placeholder.
    feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
    feed_dictionary[dropout_placeholder] = 1.0
    for target_placeholder, target_value in zip(target_placeholders, batch_target_values):
      feed_dictionary[target_placeholder] = target_value
    for c_state_placeholder, prev_batch_c_state in zip(c_state_placeholders, prev_batch_c_states):
      feed_dictionary[c_state_placeholder] = prev_batch_c_state
    for h_state_placeholder, prev_batch_h_state in zip(h_state_placeholders, prev_batch_h_states):
      feed_dictionary[h_state_placeholder] = prev_batch_h_state

    # Test over 1 batch.
    outputs_value, cost_value, state_value = sess.run((outputs, cost, state), feed_dict=feed_dictionary)
    outputs_value = np.array(outputs_value)
    testing_cost += cost_value
    test_outputs.append(outputs_value[:,0,:])
    prev_batch_c_states = [state_value[i].c for i in range(len(state_value))]
    prev_batch_h_states = [state_value[i].h for i in range(len(state_value))]
    assert not np.isnan(outputs_value[-1]).any()   
test_outputs = np.array(test_outputs)

if PLOTTING_SUPPORT:
  plt.figure(figsize=(TEST_DATA_LENGTH/20,10))
  plt.gca().set_prop_cycle(cycler('color', ['k'] + 
                                           [(1,1,w) for w in np.linspace(0.8,0.5,MP.OUTSET_CUTOFF)] + 
                                           [(1,w,w) for w in np.linspace(0.4,0,MP.BPTT_LENGTH-MP.OUTSET_CUTOFF)]))
  plot_data = np.array(test_data)
  if MP.INPUT_SIZE > 1:
    plot_data = plot_data[:,0]
  plotting_function(range(len(plot_data)), plot_data, label="test data")
  if REALIGN_OUTPUT:
    abscisses = np.tile(np.arange(MP.OUTSET_CUTOFF, MP.OUTSET_CUTOFF+len(test_outputs))[:,None], (1,MP.BPTT_LENGTH))
    abscisses = abscisses + np.arange(MP.BPTT_LENGTH)
  else:
    abscisses = np.arange(MP.OUTSET_CUTOFF, MP.OUTSET_CUTOFF+len(test_outputs))
  plotting_function(abscisses, test_outputs[:,:,0], label="prediction")
  plt.legend()
  if MP.INPUT_SIZE > 1:
    plt.figure(figsize=(TEST_DATA_LENGTH/20,10))
    plt.gca().set_prop_cycle(cycler('color', ['k'] + 
                                             [(1,1,w) for w in np.linspace(0.8,0.5,MP.OUTSET_CUTOFF)] + 
                                             [(1,w,w) for w in np.linspace(0.4,0,MP.BPTT_LENGTH-MP.OUTSET_CUTOFF)]))
    plot_data = np.array(test_data)[:,1]
    plotting_function(range(len(plot_data)), plot_data, label="electrode 1")
    plotting_function(abscisses, test_outputs[:,:,1], label="prediction")
    plt.legend() 
    plt.figure(figsize=(TEST_DATA_LENGTH/20,10))
    plot_data = np.array(test_data)[:,1:]
    plotting_function(range(len(plot_data)), plot_data, label="electrodes")
    plt.legend()
  print("Offset: ", end='')
  print(offset)
print("Testing cost: ", end='')
print(testing_cost)

#Reset test data to normal data
offset += TEST_DATA_LENGTH
if len(raw_wave2) == 0:
  test_data = raw_wave[offset:][TRAINING_DATA_LENGTH:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
else:
  if len(raw_wave3) == 0:
    test_data = raw_wave2[offset:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
  else:
    test_data = raw_wave3[offset:][:TEST_DATA_LENGTH]


# In[ ]:

if False:
  if PLOTTING_SUPPORT:
    from IPython import display
    for i in range(len(output_value)):
      plotting_function(range(10), test_data[i:i+10][:,0], label='target')
      plotting_function(range(10), output_value[i], label='prediction')
      plt.legend()
      plt.ylim([-100,100])
      plt.show()
      plt.pause(0.01)
      display.clear_output(wait=True)


# In[ ]:

## Replace test data with sine wave
if False:
  test_data = np.tile(30*np.sin(np.linspace(0,100*np.pi,TEST_DATA_LENGTH)), (MP.INPUT_SIZE, 1)).T


# In[ ]:

#Reset test data to normal data
offset += TEST_DATA_LENGTH
if len(raw_wave2) is 0:
  test_data = raw_wave[offset:][TRAINING_DATA_LENGTH:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
else:
  if len(raw_wave3) is 0:
    test_data = raw_wave2[offset:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]
  else:
    test_data = raw_wave3[offset:][:TEST_DATA_LENGTH]


# ## Hallucination

# In[ ]:

PRE_HALLUCINATION_LENGTH = 1000
HALLUCINATION_LENGTH = 500
HALLUCINATION_FUTURE = 0

from batchmaker import StatefulBatchmaker
hal_batchmaker = StatefulBatchmaker(test_data, MP.BPTT_LENGTH, MP.OUTSET_CUTOFF, 1, MP.OUTPUT_SIZE, True)
prev_batch_c_states = [np.zeros((1, MP.NUM_UNITS)) for i in range(len(state_value))]
prev_batch_h_states = [np.zeros((1, MP.NUM_UNITS)) for i in range(len(state_value))]
batch_input_values, batch_target_values = hal_batchmaker.next_batch()

hal_target = []
pre_hal_output = []
hal_output = []
for i in range(PRE_HALLUCINATION_LENGTH):
  # Assign a value to each placeholder.
  feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
  feed_dictionary[dropout_placeholder] = 1.0
  for target_placeholder, target_value in zip(target_placeholders, batch_target_values):
    feed_dictionary[target_placeholder] = target_value
  for c_state_placeholder, prev_batch_c_state in zip(c_state_placeholders, prev_batch_c_states):
    feed_dictionary[c_state_placeholder] = prev_batch_c_state
  for h_state_placeholder, prev_batch_h_state in zip(h_state_placeholders, prev_batch_h_states):
    feed_dictionary[h_state_placeholder] = prev_batch_h_state

  # Run session
  outputs_value, state_value = sess.run((outputs, state), feed_dict=feed_dictionary)

  prev_batch_c_states = [state_value[i].c for i in range(len(state_value))]
  prev_batch_h_states = [state_value[i].h for i in range(len(state_value))]

  batch_input_values, batch_target_values = hal_batchmaker.next_batch()
  hal_target.append(batch_input_values[-1])
  pre_hal_output.append(outputs_value[MP.OUTSET_CUTOFF-1])

for i in range(HALLUCINATION_LENGTH):
  # Assign a value to each placeholder.
  feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
  feed_dictionary[dropout_placeholder] = 1.0
  for target_placeholder, target_value in zip(target_placeholders, batch_target_values):
    feed_dictionary[target_placeholder] = target_value
  for c_state_placeholder, prev_batch_c_state in zip(c_state_placeholders, prev_batch_c_states):
    feed_dictionary[c_state_placeholder] = prev_batch_c_state
  for h_state_placeholder, prev_batch_h_state in zip(h_state_placeholders, prev_batch_h_states):
    feed_dictionary[h_state_placeholder] = prev_batch_h_state

  # Run session
  outputs_value, state_value = sess.run((outputs, state), feed_dict=feed_dictionary)
  hal_output.append(outputs_value[MP.OUTSET_CUTOFF-1+HALLUCINATION_FUTURE][0,0])

  prev_batch_c_states = [state_value[i].c for i in range(len(state_value))]
  prev_batch_h_states = [state_value[i].h for i in range(len(state_value))]

  #batch_input_values, batch_target_values = hal_batchmaker.next_batch()
  #batch_input_values[-1] = outputs_value[MP.OUTSET_CUTOFF-1]

  batch_input_values.pop(0)
  batch_input_values.append(outputs_value[MP.OUTSET_CUTOFF-1])

  actual_input_values, _ = hal_batchmaker.next_batch()
  hal_target.append(actual_input_values[-1])

if PLOTTING_SUPPORT:
  plt.figure(figsize=(100,10))
  plt.gca().set_prop_cycle(cycler('color', ['k'] + ['blue'] + ['red']))
  plotting_function(range(len(hal_target)), np.array(hal_target)[:,0,0], label="test data")
  plotting_function(range(len(pre_hal_output)), np.array(pre_hal_output)[:,0,0], label="pre-hallucination")
  plotting_function(range(len(pre_hal_output),len(pre_hal_output)+len(hal_output)),hal_output, label="prediction")
  plt.xlim([0,PRE_HALLUCINATION_LENGTH+len(hal_output)])
  plt.legend()


# ## You've got mail!

# In[ ]:

with open(DATA_FOLDER+"secret.txt") as file:
    secret=file.read().replace('\n', '')


# In[ ]:

if not SET_EULER_PARAMETERS:
  import smtplib
   
  server = smtplib.SMTP('smtp.gmail.com', 587)
  server.starttls()
  server.login("brainwavesdev@gmail.com", secret)
 
  msg = "Waves brained!"
  server.sendmail("brainwavesdev@gmail.com", "brainwavesdev@gmail.com", msg)
  server.quit()

  print("Mail sent.")


# In[ ]:

## Author: Daniel Dugas

