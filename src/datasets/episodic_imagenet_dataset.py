import torchvision
from typing import Any

class EpisodicImageNet(torchvision.datasets.ImageNet):
  """Override the ImageNet Dataset to support episodic batches."""

  def __init__(self, root: str, split: str = "train", **kwargs: Any) -> None:
    super().__init__(root, split, **kwargs)
    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
    self.all_targets = list(self.target_to_index.keys())
    self.all_sample_ids = list(self.target_to_index.values())[0]

