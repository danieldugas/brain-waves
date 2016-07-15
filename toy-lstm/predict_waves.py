from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from lstm import LstmParam, LstmNetwork

LEARNING_RATE = 0.1
MEM_CELL_COUNT = 20
ADD_LSTM_2 = True
n_epochs = 10
backprop_trunc_length = 10 # a.k.a sliding window size

X_START = 2000
X_LENGTH = backprop_trunc_length+100
X_SKIP = 100

PLOT_LIVE_OUTPUT = False # Slow
PLOT_LIVE_STATE = False # Slow
PLOT_WEIGHTS = True # Slow - I recommend reducing the MEM_CELL_COUNT
PLOT_WEIGHT_STATS = False
PLOT_LOSS_STATS = True
PLOT_SLIDING_WINDOW = True

DEBUG_KEEP_WINDOW_STILL = True
if DEBUG_KEEP_WINDOW_STILL:
  DEBUG_KEEP_WINDOW_STILL_POS = 0
DEBUG_RANDOM_WINDOW = False

autosave_filename = "lstm_net_autosave.pickle"

# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    def __init__(self, pred_size):
        self.w = rand_arr(-0.1, 0.1, pred_size)
        self.b = 0

    def output(self, pred):
        return np.dot(self.w, pred) + self.b

    def loss(self, pred, label):
        return (self.output(pred) - label) ** 2

    def bottom_diff(self, pred, label, node_index=None):
        """
        node_index is unused here. Useful when loss layer is a RNN with several nodes
        """
        output_error = (self.output(pred) - label)
        diff = 2 * output_error*self.w
        # Apply a little learning to oneself
        self.w -= 0.1* (1/self.w.shape[0]) * output_error * pred
        return diff


# parameters for input data dimension and lstm cell count
mem_cell_ct = MEM_CELL_COUNT
x_dim = 1
concat_len = x_dim + mem_cell_ct
lstm_param = LstmParam(mem_cell_ct, x_dim)
lstm_net = LstmNetwork(lstm_param)
if ADD_LSTM_2:
  lstm2_param = LstmParam(mem_cell_ct, mem_cell_ct) # second lstm input is first lstm state.h
  lstm2_net = LstmNetwork(lstm2_param)
## Load net if it exists
import pickle
try:
  with open(autosave_filename, "rb") as input_file:
    lstm_net = pickle.load(input_file)
  print("Loaded autosaved model")
except:
  print("No autosaved model found")
if ADD_LSTM_2:
  try:
    with open("2" + autosave_filename, "rb") as input_file:
      lstm2_net = pickle.load(input_file)
    print("Loaded autosaved model for lstm 2")
  except:
    print("No autosaved model found for lstm 2")

# Load dataset
from load_data import load_raw_waves
raw_data = load_raw_waves()
x_list = raw_data[X_START:X_START+(X_LENGTH*X_SKIP):X_SKIP]
x_list = x_list / max(abs(x_list))
# the output values are the next input values (the LSTM has to predict them)
y_list = x_list[1:]
y_list = np.append(y_list, 0)
# loss layer
loss_layer = ToyLossLayer(mem_cell_ct)

# Useful value
n_window_positions_per_epoch = (len(x_list)-backprop_trunc_length)

## Training
loss_log = []
epoch_avg_loss_log = []
avg_weight_log = []
min_weight_log = []
max_weight_log = []
for epoch in range(n_epochs):

  plt.figure('data')
  plt.suptitle("Epoch " + str(epoch) )
  # Prepare the positions the sliding window will jump through
  sliding_window_positions = range(backprop_trunc_length, len(y_list))
  if DEBUG_KEEP_WINDOW_STILL:
    sliding_window_positions = [backprop_trunc_length+DEBUG_KEEP_WINDOW_STILL_POS
                                for i in sliding_window_positions]
  if DEBUG_RANDOM_WINDOW:
    np.random.shuffle(sliding_window_positions)

  # Move a sliding window along the whole dataset. Train within the window
  for i_sw, sliding_window_position in enumerate(sliding_window_positions):
    # Create the window
    current_window_indices = range( sliding_window_position - backprop_trunc_length, sliding_window_position )
    current_window_y_list = y_list[current_window_indices]

    # Perform forward prop on whole sliding window, creating nodes as we go
    y_pred = []
    for node, index in enumerate(current_window_indices):
      lstm_net.x_list_add(x_list[index])
      if ADD_LSTM_2:
        lstm2_net.x_list_add( lstm_net.lstm_node_list[node].state.h )
        output = loss_layer.output( lstm2_net.lstm_node_list[node].state.h )
      else:
        output = loss_layer.output( lstm_net.lstm_node_list[node].state.h )
      y_pred.append( output )

      if PLOT_LIVE_STATE:
        plt.figure('live_state')
        plt.clf()
        plt.pcolor( np.reshape(lstm_net.lstm_node_list[node].state.h, (-1, 1)) )
        plt.colorbar()
        plt.tight_layout()
      if PLOT_LIVE_OUTPUT:
        plt.figure('live_output')
        if len(y_pred) == 1:
          plt.cla()
        plt.scatter( node, output )
        plt.pause(0.005)
        plt.tight_layout()

    # Display current window
    if PLOT_SLIDING_WINDOW:
      plt.figure('data')
      plt.cla()
      plt.plot(x_list)
      plt.axvline(current_window_indices[0])
      plt.axvline(current_window_indices[-1])
      plt.plot(current_window_indices, y_pred)
      plt.tight_layout()

    if PLOT_WEIGHT_STATS:
      allweights = lstm_net.all_weights()
      avg_weight_log.append( np.mean(allweights) )
      min_weight_log.append( allweights.min() )
      max_weight_log.append( allweights.max() )
      plt.figure('weight_stats', figsize=(10,2))
      plt.cla()
      plt.plot(avg_weight_log)
      plt.plot(min_weight_log)
      plt.plot(max_weight_log)
      plt.xlim([-10,n_epochs*n_window_positions_per_epoch])
      for i in range(n_epochs):
        plt.axvline(i*n_window_positions_per_epoch)
      plt.tight_layout()

    if PLOT_WEIGHTS:
      allweights = lstm_net.all_weights()
      plt.figure('weights')
      plt.clf()
      plt.pcolor( allweights )
      plt.colorbar()
      plt.tight_layout()
#       raw_input("Press any key to backprop")

    # Perform backprop on whole sliding window (backwards through nodes)
    if ADD_LSTM_2:
      loss = lstm2_net.y_list_is(current_window_y_list, loss_layer)
      # no need to feed a y_list, as lstm2_net has computed its bottom_diffs by itself
      unused = [None]*len(current_window_y_list)
      lstm_net.y_list_is(unused, lstm2_net)
    else:
      loss = lstm_net.y_list_is(current_window_y_list, loss_layer)
    lstm_param.apply_diff(lr=LEARNING_RATE)
    lstm_net.x_list_clear()
    lstm_net.lstm_node_list = []
    if ADD_LSTM_2:
      lstm2_param.apply_diff(lr=LEARNING_RATE)
      lstm2_net.x_list_clear()
      lstm2_net.lstm_node_list = []


    # Store the loss
    loss_log.append( np.sum(np.abs(loss)) )

    if PLOT_LOSS_STATS:
      plt.figure('loss_stats', figsize=(16,2))
      plt.cla()
      plt.plot(loss_log)
      x_axis_positions = (0.5+np.arange(len(epoch_avg_loss_log)))*n_window_positions_per_epoch
      plt.scatter(x_axis_positions, epoch_avg_loss_log)
      plt.xlim([-10,n_epochs*n_window_positions_per_epoch])
      plt.ylim([0, 1.1*max(loss_log)])
      for i in range(n_epochs):
        plt.axvline(i*n_window_positions_per_epoch)
      plt.tight_layout()

    plt.pause(0.005)

  epoch_avg_loss_log.append( np.mean(loss_log[-n_window_positions_per_epoch:]) )


## TEST ##
##########


# Create the window
test_indices = range(len(x_list))

# Perform forward prop
test_pred = []
for node, index in enumerate(test_indices):
  lstm_net.x_list_add(x_list[index])
  if ADD_LSTM_2:
    lstm2_net.x_list_add( lstm_net.lstm_node_list[node].state.h )
    output = loss_layer.output( lstm2_net.lstm_node_list[node].state.h )
  else:
    output = loss_layer.output( lstm_net.lstm_node_list[node].state.h )
  test_pred.append( output )

plt.figure('data')
plt.plot(test_pred)

lstm_net.x_list_clear()
lstm_net.lstm_node_list = []
if ADD_LSTM_2:
  lstm2_net.x_list_clear()
  lstm2_net.lstm_node_list = []

import pickle
with open(autosave_filename, "wb") as output_file:
  pickle.dump(lstm_net, output_file)
print("Model autosaved to " + autosave_filename)
if ADD_LSTM_2:
  with open("2" + autosave_filename, "wb") as output_file:
    pickle.dump(lstm2_net, output_file)
  print("Model 2 autosaved to 2" + autosave_filename)
