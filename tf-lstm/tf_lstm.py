
# coding: utf-8

# In[1]:

from __future__ import print_function

import tensorflow as tf
import numpy as np


# In[10]:

BATCH_SIZE = 1000
TRAINING_DATA_LENGTH = 100000
VAL_DATA_LENGTH = 10000
TEST_DATA_LENGTH = 200
SHUFFLE_TRAINING_EXAMPLES = True
SCALE_DATA = False
SAMPLING = 100

MAX_STEPS = 100000

VAL_EVERY_N_STEPS = 10
VAL_STEP_TOLERANCE = 10

BPTT_LENGTH = 100
NUM_UNITS = 128 # if this is increased, decrease the LEARNING_RATE (don't know why)
N_LAYERS = 3
INPUT_SIZE = 1
LEARNING_RATE = 0.001
CLIP_GRADIENTS = 1.0
SCALE_OUTPUT = 1000.0

SAVE_DIR = "/cluster/home/dugasd/tf-lstm-model/"
SAVE_FILE = "model.ckpt"
SAVE_PATH = SAVE_DIR+SAVE_FILE
TENSORBOARD_DIR = "./tensorboard"

DATA_FOLDER = "/cluster/home/dugasd/"
DATA_FILENAME="001_Session1_FilterTrigCh_RawCh.mat"
#DATA_FILENAME="001_Session2_FilterTrigCh_RawCh.mat"


# In[3]:

## Create Graph
tf.reset_default_graph()

with tf.name_scope("input_placeholders") as scope:
  input_placeholders = [tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name="input"+str(i))
                        for i in range(BPTT_LENGTH)]
  # shape[0] is None instead of BATCH_SIZE to allow for variable batch size

stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(NUM_UNITS, state_is_tuple=True)] * N_LAYERS , state_is_tuple=True)
unrolled_outputs, state = tf.nn.rnn(stacked_lstm, input_placeholders, dtype=tf.float32)

outputs = [tf.mul(cell_output[:, 0:1], tf.constant(SCALE_OUTPUT)) for cell_output in unrolled_outputs]

target_placeholder = tf.placeholder(tf.float32, shape=(None, INPUT_SIZE), name="target")

loss = tf.square(target_placeholder - outputs[-1], name="loss")
cost = tf.reduce_mean(loss, name="cost")   # average over batch
# Use ADAM optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
if CLIP_GRADIENTS > 0:
  adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
  gvs = adam.compute_gradients(cost)
#  capped_gvs = [(tf.clip_by_value(grad, -CLIP_GRADIENTS, CLIP_GRADIENTS), var) for grad, var in gvs]
  capped_gvs = [(tf.clip_by_norm(grad, CLIP_GRADIENTS), var) for grad, var in gvs]
  optimizer = adam.apply_gradients(capped_gvs)

# Initialize session and write graph for visualization.
sess = tf.Session()
tf.initialize_all_variables().run(session=sess)
summary_writer = tf.train.SummaryWriter(TENSORBOARD_DIR, sess.graph)

print("Session Created")


# In[4]:

saver = tf.train.Saver()


# In[11]:

import scipy.io
mat = scipy.io.loadmat(DATA_FOLDER+DATA_FILENAME)
global raw_wave
raw_wave = mat.get('data')[0]
if SCALE_DATA:
  raw_wave = raw_wave/max(raw_wave)
raw_wave = raw_wave[::SAMPLING]
raw_wave = raw_wave[0:]
assert len(raw_wave) >= TRAINING_DATA_LENGTH+TEST_DATA_LENGTH+VAL_DATA_LENGTH
training_data = raw_wave[:TRAINING_DATA_LENGTH]
val_data = raw_wave[TRAINING_DATA_LENGTH:][:VAL_DATA_LENGTH]
test_data = raw_wave[TRAINING_DATA_LENGTH:][VAL_DATA_LENGTH:][:TEST_DATA_LENGTH]

print(len(raw_wave)-TRAINING_DATA_LENGTH+TEST_DATA_LENGTH+VAL_DATA_LENGTH)


# In[6]:

# Restore model weights from previously saved model
import os
if os.path.exists(SAVE_PATH):
  saver.restore(sess, SAVE_PATH)
  print("Model restored from file: %s" % SAVE_PATH)
else:
  print("No model found.")


# ### Training

# In[7]:

from batchmaker import Batchmaker

step_cost_log = []
val_cost_log = []
val_steps_since_last_improvement = 0

# single step
for step in range(MAX_STEPS):
  # Validation
  if np.mod(step, VAL_EVERY_N_STEPS) == 0:
    val_batchmaker = Batchmaker(val_data, BPTT_LENGTH, "max", shuffle_examples=False)
    batch_input_values, batch_target_values = val_batchmaker.next_batch()
    
    # Assign a value to each placeholder.
    feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
    feed_dictionary[target_placeholder] = batch_target_values

    # Train over 1 batch.
    cost_value = sess.run(cost, feed_dict=feed_dictionary)
    print("Validation cost: ", end='')
    print(cost_value)
    val_cost_log.append(cost_value)
    
    # Check if cost has improved
    if len(val_cost_log) > 1:
        if val_cost_log[-1] <  min(val_cost_log[:-1]):
            val_steps_since_last_improvement = 0
            # save model to disk
            import os
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
                print("Created directory: %s" % SAVE_DIR)
            print("Saving ... ", end='')
            save_path = saver.save(sess, SAVE_PATH)
            print("Model saved in file: %s" % save_path)
        else:
            val_steps_since_last_improvement += 1
    # Stop training if val_cost hasn't improved in VAL_STEP_TOLERANCE steps
    if val_steps_since_last_improvement > VAL_STEP_TOLERANCE:
        print("Training stopped by validation monitor.")
        break
            
  # Train on batches
  training_batchmaker = Batchmaker(training_data, BPTT_LENGTH, BATCH_SIZE, 
                                   shuffle_examples=SHUFFLE_TRAINING_EXAMPLES)
  total_step_cost = 0
  while True:
    if training_batchmaker.is_depleted():
      break
    else:
      batch_input_values, batch_target_values = training_batchmaker.next_batch()
      
      # Assign a value to each placeholder.
      feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
      feed_dictionary[target_placeholder] = batch_target_values
  
      # Train over 1 batch.
      opt_value, last_output_value, cost_value = sess.run((optimizer, outputs[-1], cost),
                                                        feed_dict=feed_dictionary)
      total_step_cost += cost_value
      assert not np.isnan(last_output_value).any()


print("Training ended.")


# In[8]:

# Restore model weights from previously saved model
import os
if os.path.exists(SAVE_PATH):
  saver.restore(sess, SAVE_PATH)
  print("Model restored from file: %s" % SAVE_PATH)
else:
  print("No model found.")


# ### Testing

# In[12]:

from batchmaker import Batchmaker
test_batchmaker = Batchmaker(test_data, BPTT_LENGTH, "max", shuffle_examples=False)
batch_input_values, batch_target_values = test_batchmaker.next_batch()
    
# Assign a value to each placeholder.
feed_dictionary = {ph: v for ph, v in zip(input_placeholders, batch_input_values)}
feed_dictionary[target_placeholder] = batch_target_values

# Run session
cost_value, output_value = sess.run((cost, outputs[-1]), feed_dict=feed_dictionary)

print("Testing cost: ", end='')
print(cost_value)


# In[ ]:

#Replace test data with sine wave
test_data = np.sin(np.linspace(0,10*np.pi,TEST_DATA_LENGTH))


# In[ ]:

#Reset test data to normal data
offset = TEST_DATA_LENGTH
test_data = raw_wave[TRAINING_DATA_LENGTH:][VAL_DATA_LENGTH:][offset:][:TEST_DATA_LENGTH]


# In[ ]:

## Author: Daniel Dugas

