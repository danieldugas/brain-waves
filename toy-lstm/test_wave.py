x = 0

# Perform forward prop on whole sliding window, creating nodes as we go
y_pred = []
for node in range(1000):
  lstm_net.x_list_add(x)
  y_pred.append( lstm_net.lstm_node_list[node].state.h[0] )
  x = y_pred[-1]

import matplotlib.pyplot as plt
plt.ion()
plt.figure()
plt.plot(y_pred)
plt.pause(0.05)
