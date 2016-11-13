import numpy as np

class Batchmaker:
    def __init__(self, input_data, is_sleep_data, examples_per_batch, model_params, shuffle_examples=True):
        self.input_data = input_data
        self.is_sleep_data = is_sleep_data
        self.input_shape = model_params.INPUT_SHAPE
        self.target_shape = model_params.WAVE_OUT_SHAPE
        self.example_width = self.input_shape[0] + self.target_shape[0]
        # create example indices list
        self.remaining_example_indices = list(range(len(input_data) - self.example_width))
        #   shuffle list if required
        if shuffle_examples:
            from random import shuffle
            shuffle(self.remaining_example_indices)
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(self.remaining_example_indices)
        assert type(examples_per_batch) is int
        if examples_per_batch > len(self.remaining_example_indices):
            print("WARNING: more examples per batch than possible examples in all input_data")
            self.examples_per_batch = len(self.remaining_example_indices)
        else:
            self.examples_per_batch = examples_per_batch
        # initialize counter
        self.batches_consumed_counter = 0

    def next_batch(self):
        assert not self.is_depleted()
        # Create a single batch
        batch_input_values    = np.zeros([self.examples_per_batch] + self.input_shape)
        batch_target_values   = np.zeros([self.examples_per_batch] + self.target_shape)
        batch_is_sleep_values = np.zeros([self.examples_per_batch] + self.target_shape)
        for i_example in range(self.examples_per_batch):
          # Create training example at index 'pos' in input_data.
          pos = self.remaining_example_indices.pop(0)
          example_slice = list(range(pos,pos+self.example_width))
          input_slice = example_slice[:self.input_shape[0]]
          target_slice = example_slice[self.input_shape[0]:]
          # extract data
          batch_input_values[i_example] = np.reshape(self.input_data[input_slice], self.input_shape)
          batch_target_values[i_example] = np.reshape(self.input_data[target_slice], self.target_shape)
          batch_is_sleep_values[i_example] = np.reshape(self.is_sleep_data[target_slice], self.target_shape)

        self.batches_consumed_counter += 1

        return batch_input_values, batch_target_values, batch_is_sleep_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter

def progress_bar(batchmaker):
  from matplotlib import pyplot as plt  
  import time
  plt.figure('progress_bar')
  plt.scatter(time.time(), batchmaker.n_batches_consumed())
  plt.ylim([0, batchmaker.n_batches_consumed()+batchmaker.n_batches_remaining()])
  plt.show()
  plt.gcf().canvas.draw()
  time.sleep(0.0001)
