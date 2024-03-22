from collections import defaultdict
import json
import torch
from datasets import load_dataset


class WikiArt(torch.utils.data.Dataset):

  def __init__(self,
               split: str = "train",
               class_column: str = "style",
               transform=None):
    """class_column should be the attribute that we want to treat as classes. Either 'artist', 'genre', or 'style'"""
    super().__init__()
    # Kind of hacky, but wikiart only has a train split, so we'll split out
    # 8000 images (out of 81444) to serve as our validation split.
    path_to_datasets = '../image_datasets/'
    with open(f'{path_to_datasets}/wikiart/wikiart_val_indices.json') as f:
      self.val_indices = json.load(f)
      self.val_indices.sort()

    self.base_dataset = load_dataset("huggan/wikiart")
    self.transform = transform
    self.class_column = class_column
    self.base_dataset_len = len(self.base_dataset['train'])
    self.split = split

    # index map
    self.split_to_index = defaultdict(list)
    for i in range(self.base_dataset_len):
      idx_split = 'train'
      if i in self.val_indices:
        idx_split = 'val'
      self.split_to_index[idx_split].append(i)

    self.classes = self.base_dataset.unique(class_column)['train']
    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    # Get target_to_index but only for the current split
    self.target_only_dataset = self.base_dataset.select_columns(
      [class_column])

    for i, k in enumerate(self.split_to_index[split]):
      row = self.target_only_dataset['train'][k]  # Get the true training example index.
      self.target_to_index[row[class_column]].append(i)  # Map the true training example index to its numeric val.
    print(f'Loaded wikiart-{class_column} dataset ({split})')

  def __len__(self):
    return len(self.split_to_index[self.split])

  def __getitem__(self, idx):
    split_idx = self.split_to_index[self.split][idx]
    row = self.base_dataset['train'][split_idx]
    image = row['image']
    if self.transform is not None:
      image = self.transform(image)
    return image, row[self.class_column]
