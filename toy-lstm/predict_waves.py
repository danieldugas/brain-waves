from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from lstm import LstmParam, LstmNetwork

## Parameters ##
################
class DefaultParameters(object):
    """
    Change the following as much as your heart desires
    """
    def __init__(self):
      self.RAW_DATA_FOLDER = "/home/daniel/Downloads/Raw-Waves/"

      self.LEARNING_RATE = 0.1
      self.MEM_CELL_COUNT = 512
      self.ADD_LSTM_2 = True
      self.N_EPOCHS = 10
      self.BPTT_LENGTH = 100 # a.k.a sliding window size

      self.X_START = 2000
      self.X_LENGTH = self.BPTT_LENGTH+100
      self.X_SKIP = 100
      self.SCALE_X = False

      self.PLOT_LIVE_OUTPUT = False # Slow
      self.PLOT_LIVE_STATE = False # Slow
      self.PLOT_WEIGHTS = False # Slow - I recommend reducing the MEM_CELL_COUNT
      self.PLOT_WEIGHT_STATS = False
      self.PLOT_LOSS_STATS = True
      self.PLOT_SLIDING_WINDOW = True

      self.DEBUG_KEEP_WINDOW_STILL = False
      if self.DEBUG_KEEP_WINDOW_STILL:
        self.DEBUG_KEEP_WINDOW_STILL_POS = 0
      self.DEBUG_RANDOM_WINDOW = True

      self.AUTOSAVE_FILENAME = "lstm_net_autosave.pickle"
      self.AUTOSAVE_MODEL_AT_EVERY_EPOCH = False

      self.LOGPATH = "loss_log.txt"
      self.LOG_EPOCH_AVG_LOSS_INSTEAD = False

      self.TEST_AND_PLOT_PREDICTION = True

class EulerParameters(DefaultParameters):
    def __init__(self):
      # Init the default parameters
      super(EulerParameters, self).__init__()
      # Override specific ones
      #   More epochs
      self.N_EPOCHS = 100
      #   Disable all plotting
      self.PLOT_LIVE_OUTPUT = False # Slow
      self.PLOT_LIVE_STATE = False # Slow
      self.PLOT_WEIGHTS = False # Slow - I recommend reducing the MEM_CELL_COUNT
      self.PLOT_WEIGHT_STATS = False
      self.PLOT_LOSS_STATS = False
      self.PLOT_SLIDING_WINDOW = False
      #   Raw data folder
      self.RAW_DATA_FOLDER = "/cluster/home/dugasd/"
      #   Log avg loss instead (lighter log)
      self.LOG_EPOCH_AVG_LOSS_INSTEAD = True
      #   Autosave regularly
      self.AUTOSAVE_MODEL_AT_EVERY_EPOCH = True
      #   Training only
      self.TEST_AND_PLOT_PREDICTION = False

P = DefaultParameters()

## Handle Command-line arguments ##
###################################
import sys
argv = sys.argv[:]
if len(argv) > 1:
  script_path = argv.pop(0)
  if "--euler" in argv:
    P = EulerParameters()
    print("Parameters set for execution on euler cluster")
    argv.remove("--euler")
  while len(argv) > 0:
    arg = argv.pop(0)
    if arg == "-N_EPOCHS":
      P.N_EPOCHS = int(argv.pop(0))
      print("N_EPOCHS set to " + str(P.N_EPOCHS))
    elif arg == "-PLOT_WEIGHTS":
      P.PLOT_WEIGHTS = bool(argv.pop(0))
      print("PLOT_WEIGHTS set to " + str(P.PLOT_WEIGHTS))
    else:
      print("WARNING: Ignored unknown/duplicate argument '" + arg + "'")

## Useful functions and classes ##
##################################

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


## Start of Execution ##
########################
# variables for input data dimension and lstm cell count
mem_cell_ct = P.MEM_CELL_COUNT
x_dim = 1
concat_len = x_dim + mem_cell_ct
lstm_param = LstmParam(mem_cell_ct, x_dim)
lstm_net = LstmNetwork(lstm_param)
if P.ADD_LSTM_2:
  lstm2_param = LstmParam(mem_cell_ct, mem_cell_ct) # second lstm input is first lstm state.h
  lstm2_net = LstmNetwork(lstm2_param)

## Load net if it exists
import pickle
try:
  with open(P.AUTOSAVE_FILENAME, "rb") as input_file:
    lstm_net = pickle.load(input_file)
  print("Loaded autosaved model")
except:
  print("No autosaved model found")
if P.ADD_LSTM_2:
  try:
    with open("2" + P.AUTOSAVE_FILENAME, "rb") as input_file:
      lstm2_net = pickle.load(input_file)
    print("Loaded autosaved model for lstm 2")
  except:
    print("No autosaved model found for lstm 2")

# Load dataset
from load_data import load_raw_waves
raw_data = load_raw_waves(folder=P.RAW_DATA_FOLDER)
x_list = raw_data[P.X_START:P.X_START+(P.X_LENGTH*P.X_SKIP):P.X_SKIP]
if P.SCALE_X:
  x_list = x_list / max(abs(x_list))
# the output values are the next input values (the LSTM has to predict them)
y_list = x_list[1:]
y_list = np.append(y_list, 0)
# loss layer
loss_layer = ToyLossLayer(mem_cell_ct)

# Useful value
n_window_positions_per_epoch = (len(x_list)-P.BPTT_LENGTH)

## Training
from timeit import default_timer as timer
start_time = timer()
loss_log = []
epoch_avg_loss_log = []
avg_weight_log = []
min_weight_log = []
max_weight_log = []
for epoch in range(P.N_EPOCHS):

  print( "Epoch " + str(epoch) )
  if P.PLOT_SLIDING_WINDOW:
    plt.figure('data')
    plt.suptitle( "Epoch " + str(epoch) )
  # Prepare the positions the sliding window will jump through
  sliding_window_positions = range(P.BPTT_LENGTH, len(y_list))
  if P.DEBUG_KEEP_WINDOW_STILL:
    sliding_window_positions = [P.BPTT_LENGTH+P.DEBUG_KEEP_WINDOW_STILL_POS
                                for i in sliding_window_positions]
  if P.DEBUG_RANDOM_WINDOW:
    np.random.shuffle(sliding_window_positions)

  # Move a sliding window along the whole dataset. Train within the window
  for i_sw, sliding_window_position in enumerate(sliding_window_positions):
    # Create the window
    current_window_indices = range( sliding_window_position - P.BPTT_LENGTH, sliding_window_position )
    current_window_y_list = y_list[current_window_indices]

    # Perform forward prop on whole sliding window, creating nodes as we go
    y_pred = []
    for node, index in enumerate(current_window_indices):
      lstm_net.x_list_add(x_list[index])
      if P.ADD_LSTM_2:
        lstm2_net.x_list_add( lstm_net.lstm_node_list[node].state.h )
        output = loss_layer.output( lstm2_net.lstm_node_list[node].state.h )
      else:
        output = loss_layer.output( lstm_net.lstm_node_list[node].state.h )
      y_pred.append( output )

      if P.PLOT_LIVE_STATE:
        plt.figure('live_state')
        plt.clf()
        plt.pcolor( np.reshape(lstm_net.lstm_node_list[node].state.h, (-1, 1)) )
        plt.colorbar()
        plt.tight_layout()
      if P.PLOT_LIVE_OUTPUT:
        plt.figure('live_output')
        if len(y_pred) == 1:
          plt.cla()
        plt.scatter( node, output )
        plt.pause(0.005)
        plt.tight_layout()

    # Display current window
    if P.PLOT_SLIDING_WINDOW:
      plt.figure('data')
      plt.cla()
      plt.plot(x_list)
      plt.axvline(current_window_indices[0])
      plt.axvline(current_window_indices[-1])
      plt.plot(current_window_indices, y_pred)
      plt.tight_layout()

    if P.PLOT_WEIGHT_STATS:
      allweights = lstm_net.all_weights()
      avg_weight_log.append( np.mean(allweights) )
      min_weight_log.append( allweights.min() )
      max_weight_log.append( allweights.max() )
      plt.figure('weight_stats', figsize=(10,2))
      plt.cla()
      plt.plot(avg_weight_log)
      plt.plot(min_weight_log)
      plt.plot(max_weight_log)
      plt.xlim([-10,P.N_EPOCHS*n_window_positions_per_epoch])
      for i in range(P.N_EPOCHS):
        plt.axvline(i*n_window_positions_per_epoch)
      plt.tight_layout()

    if P.PLOT_WEIGHTS:
      allweights = lstm_net.all_weights()
      plt.figure('weights')
      plt.clf()
      plt.pcolor( allweights )
      plt.colorbar()
      plt.tight_layout()
      if P.ADD_LSTM_2:
        allweights2 = lstm2_net.all_weights()
        plt.figure('weights2')
        plt.clf()
        plt.pcolor( allweights2 )
        plt.colorbar()
        plt.tight_layout()

    # Perform backprop on whole sliding window (backwards through nodes)
    if P.ADD_LSTM_2:
      loss = lstm2_net.y_list_is(current_window_y_list, loss_layer)
      # no need to feed a y_list, as lstm2_net has computed its bottom_diffs by itself
      unused = [None]*len(current_window_y_list)
      lstm_net.y_list_is(unused, lstm2_net)
    else:
      loss = lstm_net.y_list_is(current_window_y_list, loss_layer)
    lstm_param.apply_diff(lr=P.LEARNING_RATE)
    lstm_net.x_list_clear()
    lstm_net.lstm_node_list = []
    if P.ADD_LSTM_2:
      lstm2_param.apply_diff(lr=P.LEARNING_RATE)
      lstm2_net.x_list_clear()
      lstm2_net.lstm_node_list = []


    # Store the loss
    loss_log.append( np.sum(np.abs(loss)) )

    if P.PLOT_LOSS_STATS:
      plt.figure('loss_stats', figsize=(16,2))
      plt.cla()
      plt.plot(loss_log)
      x_axis_positions = (0.5+np.arange(len(epoch_avg_loss_log)))*n_window_positions_per_epoch
      plt.scatter(x_axis_positions, epoch_avg_loss_log)
      plt.xlim([-10,P.N_EPOCHS*n_window_positions_per_epoch])
      plt.ylim([0, 1.1*max(loss_log)])
      for i in range(P.N_EPOCHS):
        plt.axvline(i*n_window_positions_per_epoch)
      plt.tight_layout()

    plt.pause(0.005)

  epoch_avg_loss_log.append( np.mean(loss_log[-n_window_positions_per_epoch:]) )
  ## Export Log
  with open(P.LOGPATH, 'w') as outputfile:
    if P.LOG_EPOCH_AVG_LOSS_INSTEAD:
      for value in list(epoch_avg_loss_log):
          outputfile.write(str(value) + "\n")
    else:
      for value in list(loss_log):
          outputfile.write(str(value) + "\n")
  ## Export model
  is_last_epoch = epoch == (P.N_EPOCHS - 1)
  if P.AUTOSAVE_MODEL_AT_EVERY_EPOCH or is_last_epoch:
    import pickle
    with open(P.AUTOSAVE_FILENAME, "wb") as output_file:
      pickle.dump(lstm_net, output_file)
    print("Model autosaved to " + P.AUTOSAVE_FILENAME)
    if P.ADD_LSTM_2:
      with open("2" + P.AUTOSAVE_FILENAME, "wb") as output_file:
        pickle.dump(lstm2_net, output_file)
      print("Model 2 autosaved to 2" + P.AUTOSAVE_FILENAME)



## Testing
if P.TEST_AND_PLOT_PREDICTION:
  # Create the window
  test_indices = range(len(x_list))

  # Perform forward prop
  test_pred = []
  for node, index in enumerate(test_indices):
    lstm_net.x_list_add(x_list[index])
    if P.ADD_LSTM_2:
      lstm2_net.x_list_add( lstm_net.lstm_node_list[node].state.h )
      output = loss_layer.output( lstm2_net.lstm_node_list[node].state.h )
    else:
      output = loss_layer.output( lstm_net.lstm_node_list[node].state.h )
    test_pred.append( output )

  plt.figure('data')
  plt.plot(test_pred)

  lstm_net.x_list_clear()
  lstm_net.lstm_node_list = []
  if P.ADD_LSTM_2:
    lstm2_net.x_list_clear()
    lstm2_net.lstm_node_list = []

# Timekeeping
end_time = timer()
execution_time = end_time - start_time
print("Execution time:")
print(execution_time)
