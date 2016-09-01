import numpy as np

class Batchmaker:
    def __init__(self, data, BPTT_length, examples_per_batch, output_size=1, shuffle_examples=True):
        self.data = data
        self.input_size = 1
        if len(data[0].shape) == 1:
            self.input_size = data[0].shape[0]
        self.output_size = output_size
        self.BPTT_length = BPTT_length
        self.example_length = BPTT_length + self.output_size
        assert self.example_length < len(data)
        # examples per batch
        if examples_per_batch is "max":
            examples_per_batch = len(data) - self.example_length
        assert type(examples_per_batch) is int
        if examples_per_batch > len(data) - self.example_length:
            print("WARNING: more examples per batch than possible examples in all data")
            self.examples_per_batch = len(data) - self.example_length
        else:
            self.examples_per_batch = examples_per_batch
        # initialize example indices list
        self.remaining_example_indices = list(range(len(data)-self.example_length))
        # shuffle list if required
        if shuffle_examples:
            from random import shuffle
            shuffle(self.remaining_example_indices)

    def next_batch(self):
        # Create a single batch
        batch_input_values = [np.zeros((self.examples_per_batch, self.input_size)) for _ in range(self.BPTT_length)]
        batch_target_values = np.zeros((self.examples_per_batch, self.output_size))
        for i_example in range(self.examples_per_batch):
          # Create training example at index 'pos' in data.
          pos = self.remaining_example_indices.pop(0)
          #   input.
          unrolled_input_data = self.data[pos:pos+self.BPTT_length]
          for t, value in enumerate(unrolled_input_data):
              batch_input_values[t][i_example, :] = value
          #   target.
          unrolled_target_data = self.data[pos+self.BPTT_length:][:self.output_size]
          if self.input_size > 1:
              batch_target_values[i_example, :] = np.array([electrodes[0] for electrodes in unrolled_target_data])
          else:
              batch_target_values[i_example, :] = np.array([electrode for electrode in unrolled_target_data])

        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch
