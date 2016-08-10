import tensorflow as tf
import numpy as np

BPTT_length = 100
num_units = 512
layers = 2
batch_size = 10
input_size = 1
learning_rate = 0.03

## Create Graph
with tf.name_scope("input_placeholders") as scope:
  input_placeholders = [tf.placeholder(tf.float64, shape=(batch_size, input_size), name="input"+str(i))
                        for i in range(BPTT_length)]

stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(
    [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)] * layers , state_is_tuple=True)
unrolled_outputs, state = tf.nn.rnn(stacked_lstm, input_placeholders, dtype=tf.float64)

outputs = [cell_output[:, 0:1] for cell_output in unrolled_outputs]

target_placeholder = tf.placeholder(tf.float64, shape=(batch_size, input_size), name="target")

loss = tf.abs(target_placeholder - outputs[-1], name="loss")
cost = tf.reduce_mean(loss, name="cost")   # average over batch
# Use ADAM optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
  tf.initialize_all_variables().run()

  input_values =[np.random.randn(batch_size, input_size)] * BPTT_length
  target_value = np.random.randn(batch_size, input_size)

  # Assign a value to each placeholder.
  feed_dictionary = {ph: v for ph, v in zip(input_placeholders, input_values)}
  feed_dictionary[target_placeholder] = target_value

  # Train over 1 batch.
  opt_value, last_output_value, cost_value = sess.run((optimizer, outputs[-1], cost),
                                                      feed_dict=feed_dictionary)

  summary_writer = tf.train.SummaryWriter('log_simple_graph', sess.graph)
