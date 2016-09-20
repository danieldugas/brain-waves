import numpy as np

try:
    xrange
except NameError:
    xrange = range

class Batchmaker:
    def __init__(self, data, BPTT_length, examples_per_batch, output_size=1, shuffle_examples=True):
        self.data = data
        self.input_size = 1
        if type(data) == np.ndarray:
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
        self.batches_consumed_counter = 0

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

        self.batches_consumed_counter += 1

        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.remaining_example_indices) < self.examples_per_batch

    def n_batches_remaining(self):
        return len(self.remaining_example_indices) / self.examples_per_batch

    def n_batches_consumed(self):
        return self.batches_consumed_counter

# Behaves like a circshifted list of indices.
class RemainingIndices:
    def __init__(self, data_len, start_pos):
        self.data_len = data_len
        self.start_pos = start_pos
        self.popped = 0
    def pop(self, pos):
        if (pos != 0):
          raise NotImplementedError
        self.popped += 1
    def __len__(self):
        return max(0, self.data_len - self.popped)
    def __getitem__(self, slce):
        if isinstance(slce, slice):
            start = 0 if slce.start is None else slce.start
            stop = len(self) if slce.stop is None else min(len(self), slce.stop)
            step = 1 if slce.step is None else slce.step
            return [(self.start_pos + self.popped + i) % self.data_len for i in range(start, stop, step)]
        else:
            return (self.start_pos + self.popped + slce) % self.data_len


class StatefulBatchmaker(Batchmaker):
    def __init__(self, data, BPTT_length, outset_cutoff,
                 examples_per_batch, output_size, first_example_starts_at_zero=False):
        self.data = data
        self.input_size = 1
        if type(data) == np.ndarray:
            self.input_size = data[0].shape[0]
        self.output_size = output_size
        self.BPTT_length = BPTT_length
        self.outset_cutoff = outset_cutoff
        self.example_length = BPTT_length + self.output_size
        assert self.example_length < len(data)
        self.examples_per_batch = examples_per_batch
        if examples_per_batch > len(data) - self.example_length:
            print("WARNING: more examples per batch than possible examples in all data")
            self.examples_per_batch = len(data) - self.example_length
        # initialize example indices list
        from random import sample
        self.batch_start_indices = sample(xrange(len(data)-self.example_length), self.examples_per_batch)
        if first_example_starts_at_zero:
            self.batch_start_indices[0] = 0
        self.batch_remaining_indices = [RemainingIndices(len(data), n) for n in self.batch_start_indices]
        self.batches_consumed_counter = 0

    def next_batch(self):
        assert not self.is_depleted()
        # Create a single batch
        batch_input_values  = [np.zeros((self.examples_per_batch, self.input_size)) for _ in range(self.outset_cutoff)]
        batch_target_values = [np.zeros((self.examples_per_batch, self.output_size)) for _ in range(self.outset_cutoff, self.BPTT_length)]
        # Create training example at index 'pos' in data.
        for i_example, example_remaining_indices in enumerate(self.batch_remaining_indices):
          #   input.
          unrolled_input_indices = example_remaining_indices[:self.outset_cutoff]
          unrolled_input_data = [self.data[i] for i in unrolled_input_indices]
          for t, value in enumerate(unrolled_input_data):
              batch_input_values[t][i_example, :] = value
          #   target.
          unrolled_target_indices = example_remaining_indices[self.outset_cutoff:self.BPTT_length]
          unrolled_target_data = [self.data[i] for i in unrolled_target_indices]
          for t, value in enumerate(unrolled_target_data):
              batch_target_values[t][i_example, :] = value
          #   pop the example from remaining indices
          example_remaining_indices.pop(0)

        self.batches_consumed_counter += 1
        return batch_input_values, batch_target_values

    def is_depleted(self):
        return len(self.batch_remaining_indices[0]) < self.example_length

    def n_batches_remaining(self):
        return len(self.batch_remaining_indices[0])-self.example_length

    def n_batches_consumed(self):
        return self.batches_consumed_counter

# ipython notebook example
# if False:
#     def test_batchmaker():
#         from batchmaker import StatefulBatchmaker
#         bm = StatefulBatchmaker(val_data, MP.BPTT_LENGTH, MP.OUTSET_CUTOFF, BATCH_SIZE, MP.OUTPUT_SIZE)
#         from IPython import display
#         for i in range(50):
#             for i in range(1):
#                 inpt, trgt = bm.next_batch()
#             print(bm.n_batches_remaining())
#             plotting_function(range(len(inpt)), np.array(inpt)[:,:5,0])
#             plotting_function(np.arange(len(trgt))+len(inpt), np.array(trgt)[:,:5,0])
#             plt.show()
#             plt.pause(0.01)
#             display.clear_output(wait=True)
#     test_batchmaker()
