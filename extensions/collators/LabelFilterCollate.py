from typing import List

import torch


class LabelFilterCollate:
  def __init__(self, labels:List[str], classes: List[str]):
    self.accepted_label_indexes = list(LabelFilterCollate.label_to_index(classes, c) for c in labels)

  @staticmethod
  def label_to_index(classes: List[str], label: str) -> int or None:
    idx: int = 0
    for cl in classes:
      if cl == label:
        return idx
      else:
        idx = idx + 1

    return None

  def collate_fn(self, batch):
    new_batch = None
    for el in batch:
      inputs, idx_label = el
      if len(list(idx for idx in self.accepted_label_indexes if idx_label == idx)) > 0:
        if new_batch is None:
          new_batch = [inputs.unsqueeze(0), torch.tensor(idx_label).unsqueeze(0)]
        else:
          new_batch[0] = torch.cat([new_batch[0], inputs.unsqueeze(0)])
          new_batch[1] = torch.cat([new_batch[1], torch.tensor(idx_label).unsqueeze(0)])

    return new_batch if new_batch is not None else []
