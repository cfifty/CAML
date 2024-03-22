# Adapted from https://github.com/Tsingularity/FRN.

import sys
import torch
import torchvision.datasets as datasets
from torch.utils.data import Sampler
from PIL import Image

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.evaluation.datasets import samplers, transform_manager


def get_dataset(data_path, is_training, transform_type, pre):
  dataset = datasets.ImageFolder(
    data_path,
    loader=lambda x: image_loader(path=x, is_training=is_training, transform_type=transform_type, pre=pre))

  return dataset


def meta_test_dataloader(data_path, way, shot, pre, transform_type=None, query_shot=16, trial=1000):
  dataset = get_dataset(data_path=data_path, is_training=False, transform_type=transform_type, pre=pre)

  loader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=samplers.random_sampler(data_source=dataset, way=way, shot=shot, query_shot=query_shot, trial=trial),
    num_workers=3,
    pin_memory=False)

  return loader


def image_loader(path, is_training, transform_type, pre):
  p = Image.open(path)
  p = p.convert('RGB')

  if type(transform_type) == type(50):
    final_transform = transform_manager.get_transform(is_training=is_training, transform_type=transform_type, pre=pre)
  else:
    final_transform = transform_type

  p = final_transform(p)

  return p
