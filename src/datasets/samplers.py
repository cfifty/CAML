import random
import sys
import copy
import torch

from torch.utils.data import Sampler
from typing import Iterator, List

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


def custom_collate_batch_fn(batch):
  """Uses vector operations rather than a for loop over a set -- much better."""
  features, labels = [], []
  for (f, l) in batch:
    features.append(f)
    labels.append(torch.tensor(l, dtype=torch.int32))
  features = torch.stack(features, dim=0)
  labels = torch.stack(labels, dim=0)

  normed_features = torch.norm(features.reshape(len(batch), -1), dim=1)
  bool_vec = (normed_features == normed_features.unsqueeze(1))
  mask = torch.sum(bool_vec, dim=-1)  # if mask = 1, then the only True element is the diagonal.
  features = features[mask == 1]
  labels = labels[mask == 1]
  return features, labels


class MetricSampler(Sampler[int]):
  r"""Sampling for ProtoTypical Network and other classical episodic networks.

  Samples a support set as the first k elements, and n other elements belonging to the query set.

  I.e. [85, h, w, c] with way = 5 and k = 1 => the [:5, h,w,c] is the context and [5:, h,w,c] are the query points.

  ****Assumes fixed batch size of 512****
  """

  def __init__(self, num_classes, target_to_index, batch_size, way, shot) -> None:
    super().__init__(data_source=None)
    self.target_to_index = target_to_index

    # TODO(cfifty): This is actually impossible/extremely difficult to calculate due to randomness.
    self.dataset_length = 0
    query_per_class = (batch_size - way * shot) // way
    for class_id in target_to_index:
      # Number of times this shows up.
      self.dataset_length += len(target_to_index[class_id]) // (shot + query_per_class)
      if len(target_to_index[class_id]) % (shot + query_per_class) - shot >= 5:
        self.dataset_length += 1
    self.dataset_length = self.dataset_length // way  # We have |way| in each batch.

    self.shot = shot
    self.way = way
    self.num_classes = num_classes
    self.batch_size = batch_size
    if (self.batch_size - self.way * self.shot) % self.way != 0:
      raise Exception(f'Batch size does not evenly divide into way*shot samples: '
                      + f'{(self.batch_size - self.way * self.shot) % self.way} remainder -- this needs to be 0.')

  def __iter__(self) -> Iterator[int]:
    targets_dict = copy.deepcopy(self.target_to_index)
    query_per_class = (self.batch_size - self.way * self.shot) // self.way
    min_query_per_class = min(query_per_class, 5)  # Set the minimum # of query examples to be 5 per class.
    for target in targets_dict:
      random.shuffle(targets_dict[target])
    indices = []
    while targets_dict:

      # If we have fewer than |way| targets left in our targets dict, then it's time to exit.
      # Not technically necessary, since this is handled below, but nice-to-do all the same.
      targets_list = [target for target in targets_dict]
      if len(targets_dict.keys()) < self.way:
        break
      random.shuffle(targets_list)

      # Reset the "batch" that we're forming. Any leftovers from the inner loop will be discarded.
      count = 0
      query_indices = []
      support_indices = []

      for target in targets_list:
        if count < self.way:
          num_remaining = len(targets_dict[target]) - self.shot
          if num_remaining >= min_query_per_class:
            support_indices += targets_dict[target][:self.shot]
            query_indices += targets_dict[target][self.shot:self.shot + min(num_remaining, query_per_class)]
            # Pad the batch with the last-seen example.
            if num_remaining < query_per_class:
              query_indices += [targets_dict[target][-1] for _ in range(query_per_class - num_remaining)]
            targets_dict[target] = targets_dict[target][self.shot + min(num_remaining, query_per_class):]
            count += 1
          else:
            del targets_dict[target]
        elif count == self.way:
          indices += support_indices + query_indices
          query_indices = []
          support_indices = []
          count = 0
    return iter(indices)

  def __len__(self) -> int:
    return self.dataset_length
