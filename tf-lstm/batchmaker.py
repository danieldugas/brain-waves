import numpy as np

class Batchmaker:
    def __init__(self, data, BPTT_length, examples_per_batch, input_size=1, shuffle_examples=True):
        self.data = data
        self.input_size = input_size
        if input_size != 1:
            raise NotImplementedError
        self.BPTT_length = BPTT_length
        self.example_length = BPTT_length + 1
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
        batch_target_values = np.zeros((self.examples_per_batch, self.input_size))
        for example in range(self.examples_per_batch):
          # Create training example at index i in data.
          i = self.remaining_example_indices.pop(0)
          unrolled_data = self.data[i:i+self.example_length]
          for t, value in enumerate(unrolled_data[:-1]):
              batch_input_values[t][example, :] = value
          batch_target_values[example, :] = unrolled_data[-1]

        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch
