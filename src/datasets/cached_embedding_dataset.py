import os
import torch
import torchvision

import numpy as np
import torch.nn.functional as F

class CachedEmbeddingDataset(torchvision.datasets.DatasetFolder):
  """Dataset where the stored files are precomputed image embeddings.
  Expected dir structure is

  <root dir>/
      <split name>/
          <class_idx_int>/
              <image1>.npy
              <image2>.npy
              <image3>.npy
  """

  def __init__(self,
               root: str,
               split: str = "train",
               normalize_embedding=False):
    self.split = split
    root_dir = os.path.join(root, split)
    self.normalize_embedding = normalize_embedding
    if self.normalize_embedding:
      mean_embedding_file = os.path.join(root, "train",
                                         "split_average.npy")
      self.mean_embedding = torch.from_numpy(
        np.load(mean_embedding_file))

    def _loader(path: str) -> torch.Tensor:
      embedding = torch.from_numpy(np.load(path))
      # This is used for MetaQDA: https://github.com/Open-Debin/Bayesian_MQDA/blob/8da65af0b0f176b0c494e0289f9cabaf60186812/code_models/layers.py#L10
      if self.normalize_embedding and self.mean_embedding is not None:
        embedding = F.normalize(
          embedding - self.mean_embedding, p=2, dim=0
        )  # dim 0 here since we're normalizing a 1-D tensor (one at a time)
      return embedding

    super().__init__(
      root_dir,
      _loader,
      extensions=['npy'],
    )
    self.imgs = self.samples

    # turn the classes into integers
    self.classes = [int(c) for c in self.classes]
    self.classes.sort()

    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
