from __future__ import print_function

import numpy as np

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

# parameters for input data dimension and lstm cell count
mem_cell_ct = 256
x_dim = 1
concat_len = x_dim + mem_cell_ct
lstm_param = LstmParam(mem_cell_ct, x_dim)
lstm_net = LstmNetwork(lstm_param)
from load_data import load_raw_waves
raw_data = load_raw_waves()
x_list = raw_data[2000:102000:10]
# the output values are the next input values (the LSTM has to predict them)
y_list = x_list[1:]
y_list = np.append(y_list, 0)

import matplotlib.pyplot as plt
plt.ion()
plt.figure()
plt.plot(x_list)
plt.pause(0.05)

## Training
n_epochs = 10
backprop_trunc_length = 100 # a.k.a sliding window size
print("|                                                                                                    |")
print("|", end="")
epoch_total_loss = []
for epoch in range(n_epochs):

  # Prepare the positions the sliding window will jump through
  sliding_window_positions = range(backprop_trunc_length, len(y_list))
#   np.random.shuffle(sliding_window_positions)

  # Move a sliding window along the whole dataset. Train within the window
  sliding_window_total_loss = []
  for i_sw, sliding_window_position in enumerate(sliding_window_positions):
    # Create the window
    current_window_indices = range( sliding_window_position - backprop_trunc_length, sliding_window_position )
    current_window_y_list = y_list[current_window_indices]

    # Perform forward prop on whole sliding window, creating nodes as we go
    y_pred = []
    for node, index in enumerate(current_window_indices):
      lstm_net.x_list_add(x_list[index])
      y_pred.append( lstm_net.lstm_node_list[node].state.h[0] )

    # Perform backprop on whole sliding window (backwards through nodes)
    loss = lstm_net.y_list_is(current_window_y_list, ToyLossLayer)
    lstm_param.apply_diff(lr=0.1)
    lstm_net.x_list_clear()

    # Store the loss
    sliding_window_total_loss.append( np.sum(np.abs(loss)) )

    # display percentage
    if np.mod(100.0*i_sw/len(sliding_window_positions), 1) == 0:
      print(".", end="")

#     print(sliding_window_position)
#     print(y_pred)

  epoch_total_loss.append( np.sum(np.abs(sliding_window_total_loss)) )
