'''
Fills a new Batch until the batch_size is reached
useful if a collate_fn in a dataloder !!removes!! elements by returning an empty array [].

Eg.:
Input data is given as below
([1, 'lbl1'], [7, 'lbl1'], [] ,[3, 'lbl1'], [2, 'lbl2'], [], [4, 'lbl2'])
if batch_size is 2
HoneyBatcher will return following mini batches per iteration:
[1, 'lbl1'], [7, 'lbl1']
[3, 'lbl1'], [2, 'lbl2']
[4, 'lbl2']
'''
import time

import torch


class HoneyBatcher:
  def __init__(self, batch_size: int, iter_data, start: int = 0):
    self.num = start
    self.batch_size = batch_size
    self.iter_data = iter(iter_data)

  def restart(self):
    self.num = 0

  def at(self, num:int):
    self.num = num

  def __iter__(self):
    return self

  # StopIteration is raised internally
  def __next__(self):
    new_batch = None

    start_time = time.time()

    for _, next_el in enumerate(self.iter_data, self.num):
      self.num = self.num + 1
      if len(next_el) == 0:
        continue

      if new_batch is None:
        new_batch = [next_el[0], next_el[1]]
      else:
        new_batch[0] = torch.cat([new_batch[0], next_el[0]])
        new_batch[1] = torch.cat([new_batch[1], next_el[1]])

      if new_batch[0].shape[0] == self.batch_size:
        print("--- %s seconds ---" % (time.time() - start_time))
        return new_batch

    print("--- %s seconds ---" % (time.time() - start_time))
    return new_batch
