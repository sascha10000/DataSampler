from torch.utils.data.sampler import Sampler
import torch

# The code is completly taken from torch.utils.data.sampler.RandomSampler
# only minor changes are made to gain control over the randomization process
class RandomizeOnceSampler(Sampler):
  r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
  If with replacement, then user can specify :attr:`num_samples` to draw.
  Arguments:
      data_source (Dataset): dataset to sample from
      replacement (bool): samples are drawn with replacement if ``True``, default=``False``
      num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
          is supposed to be specified only when `replacement` is ``True``.
  """

  def __init__(self, data_source, num_samples=None):
    self.data_source = data_source
    self._num_samples = num_samples
    self.permutation = torch.randperm(len(self.data_source)).tolist()
    self.pos = 0

    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integer "
                       "value, but got num_samples={}".format(self.num_samples))

  @property
  def num_samples(self):
    # dataset size might change at runtime
    if self._num_samples is None:
      return len(self.data_source)
    return self._num_samples

  def __iter__(self):
    return iter(self.permutation)

  def __len__(self):
    return self.num_samples
