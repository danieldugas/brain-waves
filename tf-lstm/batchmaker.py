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
        assert not self.is_depleted()
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

def circShift(arr, n):
        return arr[n::] + arr[:n:]

class StatefulBatchmaker(Batchmaker):
    def __init__(self, data, BPTT_length, examples_per_batch, output_size, first_example_starts_at_zero=False):
        self.data = data
        self.input_size = 1
        if len(data[0].shape) == 1:
            self.input_size = data[0].shape[0]
        self.output_size = output_size
        self.BPTT_length = BPTT_length
        self.example_length = BPTT_length + self.output_size
        assert self.example_length < len(data)
        self.examples_per_batch = examples_per_batch
        if examples_per_batch > len(data) - self.example_length:
            print("WARNING: more examples per batch than possible examples in all data")
            self.examples_per_batch = len(data) - self.example_length
        # initialize example indices list
        possible_start_indices = list(range(len(data)-self.example_length))
        from random import shuffle
        shuffle(possible_start_indices)
        self.batch_start_indices = possible_start_indices[:self.examples_per_batch]
        if first_example_starts_at_zero:
            self.batch_start_indices[0] = 0
        self.batch_remaining_indices = [circShift(list(range(len(data))), n) for n in self.batch_start_indices]

    def next_batch(self):
        assert not self.is_depleted()
        # Create a single batch
        batch_input_values = [np.zeros((self.examples_per_batch, self.input_size)) for _ in range(self.BPTT_length)]
        batch_target_values = np.zeros((self.examples_per_batch, self.output_size))
        # Create training example at index 'pos' in data.
        for i_example, example_remaining_indices in enumerate(self.batch_remaining_indices):
          #   input.
          unrolled_input_indices = example_remaining_indices[:self.BPTT_length]
          unrolled_input_data = [self.data[i] for i in unrolled_input_indices]
          for t, value in enumerate(unrolled_input_data):
              batch_input_values[t][i_example, :] = value
          #   target.
          unrolled_target_indices = example_remaining_indices[self.BPTT_length:][:self.output_size]
          unrolled_target_data = [self.data[i] for i in unrolled_target_indices]
          if self.input_size > 1:
              batch_target_values[i_example, :] = np.array([electrodes[0] for electrodes in unrolled_target_data])
          else:
              batch_target_values[i_example, :] = np.array([electrode for electrode in unrolled_target_data])
          #   pop the example from remaining indices
          example_remaining_indices.pop(0)

        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.batch_remaining_indices[0]) < self.example_length

    def n_batches_remaining(self):
        return len(self.batch_remaining_indices[0])-self.example_length

# def test_batchmaker():
#     from batchmaker import StatefulBatchmaker
#     bm = StatefulBatchmaker(val_data, MP.BPTT_LENGTH, BATCH_SIZE, MP.OUTPUT_SIZE)
#
#     for i in range(50):
#           inpt, trgt = bm.next_batch()
#           print(bm.n_batches_remaining())
#           plotting_function(range(len(inpt)), np.array(inpt)[:,:5,0])
#           plotting_function(np.arange(10)+len(inpt), np.array(trgt)[:5,:].T)
