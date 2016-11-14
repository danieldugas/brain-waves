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
          input_slice = slice(pos, pos+self.input_shape[0])
          target_slice = slice(pos+self.input_shape[0], pos+self.example_width)
          # extract data
          batch_input_values[i_example]    = self.input_data[input_slice]
          batch_target_values[i_example]   = self.input_data[target_slice]
          batch_is_sleep_values[i_example] = self.is_sleep_data[target_slice]

        self.batches_consumed_counter += 1

        return batch_input_values, batch_target_values, batch_is_sleep_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter

def progress_bar(batchmaker):
  path = '/tmp/training_log.csv'
  if batchmaker == 'reset':
    from os import remove
    remove(path)
    return 0
  import time
  with open(path, 'a') as file:
    file.write(str(time.time()) + ' ' + str(batchmaker.n_batches_consumed()) + ' ' +
               str(batchmaker.n_batches_consumed()+batchmaker.n_batches_remaining()) + '\n')
  return path
