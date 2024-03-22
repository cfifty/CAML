from collections import defaultdict
import json
import torch
import tqdm
import os
from torchvision.datasets import ImageFolder

from PIL import Image

from torchvision.datasets.coco import CocoDetection


class CocoDataset(ImageFolder):

  def __init__(self, split: str = "train", transform=None):
    assert split in ['train', 'val']
    path_to_datasets = '../image_datasets/'
    super().__init__(f'{path_to_datasets}mscoco/{split}_images', transform=transform)
    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
    self.all_targets = list(self.target_to_index.keys())
    self.all_sample_ids = list(self.target_to_index.values())[0]


class CocoDetectionDataset(torch.utils.data.Dataset):

  def __init__(self,
               split: str = "train",
               transform=None):
    """class_column should be the attribute that we want to treat as classes. Either 'artist', 'genre', or 'style'"""
    super().__init__()
    assert split in ['train', 'val']
    path_to_datasets = '../image_datasets/'
    self.dataset = CocoDetection(root=f'{path_to_datasets}mscoco/{split}2017',
                                 annFile=f'{path_to_datasets}mscoco/annotations/instances_{split}2017.json',
                                 transform=transform)

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset.__getitem__(idx)


def gen_imagefolder_dataset(split):
  """
  Convert the CocoDetection Segmentation Dataset to an ImageFolder classification dataset.
  """
  dataset = CocoDetectionDataset(split)
  save_path = f'../image_datasets/mscoco/{split}_images'
  for x in dataset:
    img, labels = x
    labels_set = []
    id_set = []
    for labels_dict in labels:
      label = labels_dict['category_id']
      # In case there are multiple segmentations with the same category.
      if label not in labels_set:
        labels_set.append(label)
        id_set.append(labels_dict['id'])
    for label, id in zip(labels_set, id_set):
      os.makedirs(f'{save_path}/{label}', exist_ok=True)
      img.save(f'{save_path}/{label}/{id}.jpg')


if __name__ == '__main__':
  # gen_imagefolder_dataset('train')
  dataset = CocoDataset('train')
  for x in dataset:
    print(x)
    break
