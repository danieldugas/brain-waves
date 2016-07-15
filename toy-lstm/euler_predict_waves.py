from __future__ import print_function

import numpy as np

from lstm import LstmParam, LstmNetwork

LEARNING_RATE = 0.1
MEM_CELL_COUNT = 256
n_epochs = 1000
backprop_trunc_length = 10 # a.k.a sliding window size

PLOT_LIVE_OUTPUT = False # Slow
PLOT_LIVE_STATE = False # Slow
PLOT_WEIGHTS = False # Slow
PLOT_WEIGHT_STATS = False
PLOT_LOSS_STATS = False
PLOT_SLIDING_WINDOW = False

DEBUG_KEEP_WINDOW_STILL = False
DEBUG_RANDOM_WINDOW = False

autosave_filename = "lstm_net_autosave.pickle"
logpath = "loss_log.txt"

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

    def bottom_diff(self, pred, label):
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
## Load net if it exists
import pickle
try:
  with open(autosave_filename, "rb") as input_file:
    lstm_net = pickle.load(input_file)
  print("Loaded autosaved model")
except:
  print("No autosaved model found")

# Load dataset
from load_data import load_raw_waves
raw_data = load_raw_waves(folder="")
x_list = raw_data[2000:12000:100]
# the output values are the next input values (the LSTM has to predict them)
y_list = x_list[1:]
y_list = np.append(y_list, 0)
# loss layer
loss_layer = ToyLossLayer(mem_cell_ct)

## Training
loss_log = []
avg_weight_log = []
min_weight_log = []
max_weight_log = []
print("|                                                                                                    |")
print("|", end="")
for epoch in range(n_epochs):

  # Prepare the positions the sliding window will jump through
  sliding_window_positions = range(backprop_trunc_length, len(y_list))
  if DEBUG_KEEP_WINDOW_STILL:
    sliding_window_positions = [30 for i in sliding_window_positions]
  if DEBUG_RANDOM_WINDOW:
    np.random.shuffle(sliding_window_positions)

  # Move a sliding window along the whole dataset. Train within the window
  for i_sw, sliding_window_position in enumerate(sliding_window_positions):
    # Create the window
    current_window_indices = range( sliding_window_position - backprop_trunc_length, sliding_window_position )
    current_window_y_list = y_list[current_window_indices]

    # Display current window
    if PLOT_SLIDING_WINDOW:
      plt.figure('data')
      plt.cla()
      plt.plot(x_list)
      plt.axvline(current_window_indices[0])
      plt.axvline(current_window_indices[-1])

    ## DEBUG PLOT
    if PLOT_LIVE_OUTPUT:
      plt.figure('live_output')
      plt.cla()

    # Perform forward prop on whole sliding window, creating nodes as we go
    y_pred = []
    for node, index in enumerate(current_window_indices):
      lstm_net.x_list_add(x_list[index])
      output = loss_layer.output( lstm_net.lstm_node_list[node].state.h )
      y_pred.append( output )

      if PLOT_LIVE_STATE:
        plt.figure('live_state')
        plt.pcolor( np.reshape(lstm_net.lstm_node_list[node].state.h, (16, 16)) )
        plt.tight_layout()
      if PLOT_LIVE_OUTPUT:
        plt.figure('live_output')
        plt.scatter( node, output )
        plt.pause(0.005)
        plt.tight_layout()

    if PLOT_SLIDING_WINDOW:
      plt.figure('data')
      plt.plot(current_window_indices, y_pred)
      plt.pause(0.005)
      plt.tight_layout()

    if PLOT_WEIGHT_STATS:
      allweights = np.hstack( [lstm_net.lstm_param.wg,
                               lstm_net.lstm_param.wi,
                               lstm_net.lstm_param.wo,
                               lstm_net.lstm_param.wf,
                               lstm_net.lstm_param.bg[:,None],
                               lstm_net.lstm_param.bi[:,None],
                               lstm_net.lstm_param.bo[:,None],
                               lstm_net.lstm_param.bf[:,None]] )
      avg_weight_log.append( np.mean(allweights) )
      min_weight_log.append( allweights.min() )
      max_weight_log.append( allweights.max() )
      plt.figure('weight_stats', figsize=(10,2))
      plt.cla()
      plt.plot(avg_weight_log)
      plt.plot(min_weight_log)
      plt.plot(max_weight_log)
      plt.xlim([0,n_epochs*(len(x_list)-backprop_trunc_length)])
      for i in range(n_epochs):
        plt.axvline(i*(len(x_list)-backprop_trunc_length))
      plt.tight_layout()

    if PLOT_WEIGHTS:
      allweights = np.hstack( [lstm_net.lstm_param.wg,
                               lstm_net.lstm_param.wi,
                               lstm_net.lstm_param.wo,
                               lstm_net.lstm_param.wf,
                               lstm_net.lstm_param.bg[:,None],
                               lstm_net.lstm_param.bi[:,None],
                               lstm_net.lstm_param.bo[:,None],
                               lstm_net.lstm_param.bf[:,None]] )
      plt.figure('weights')
      plt.clf()
      plt.pcolor( allweights )
      plt.colorbar()
      plt.tight_layout()
#       raw_input("Press any key to backprop")

    # Perform backprop on whole sliding window (backwards through nodes)
    loss = lstm_net.y_list_is(current_window_y_list, loss_layer)
    lstm_param.apply_diff(lr=LEARNING_RATE)
    lstm_net.x_list_clear()
    lstm_net.lstm_node_list = []

    # Store the loss
    loss_log.append( np.sum(np.abs(loss)) )

    if PLOT_LOSS_STATS:
      plt.figure('loss_stats', figsize=(16,2))
      plt.cla()
      plt.plot(loss_log)
      plt.xlim([0,n_epochs*(len(x_list)-backprop_trunc_length)])
      plt.ylim([0, max(loss_log)])
      for i in range(n_epochs):
        plt.axvline(i*(len(x_list)-backprop_trunc_length))
      plt.tight_layout()


lstm_net.x_list_clear()
lstm_net.lstm_node_list = []

import pickle
with open(autosave_filename, "wb") as output_file:
  pickle.dump(lstm_net, output_file)
print("Model autosaved to " + autosave_filename)

with open(logpath, 'w') as outputfile:
    for value in list(loss_log):
        outputfile.write(str(value) + "\n")