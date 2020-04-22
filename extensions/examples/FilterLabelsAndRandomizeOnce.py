import sys
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List

from extensions.batch_collectors.HoneyBatcher import HoneyBatcher
from extensions.collators.LabelFilterCollate import LabelFilterCollate
from extensions.sampler.RandomizeOnceSampler import RandomizeOnceSampler

if __name__ == '__main__':
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

  collate = LabelFilterCollate(['ship', 'truck', 'automobile'], train_set.classes)

  train_data = torch.utils.data.DataLoader(train_set, num_workers=4, sampler=RandomizeOnceSampler(train_set), collate_fn=collate.collate_fn) #,

  k_folds = 3

  for k in range(0, k_folds):
    data_iterator = HoneyBatcher(batch_size=5, iter_data=train_data)
    print("Fold: " + str(k))
    for i, data in enumerate(data_iterator):
      if len(data) == 0:
        continue

      inputs, labels = data
      print(str(i) + " " + str(list(train_data.dataset.classes[label] for label in labels)))

      if i >= 10:
        break

  sys.exit()
