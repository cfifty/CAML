import torch
import sys

from torch.utils.data import Sampler
from typing import Any
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.datasets.episodic_imagenet_dataset import EpisodicImageNet
from src.datasets.cached_embedding_dataset import CachedEmbeddingDataset
from src.datasets.samplers import MetricSampler, custom_collate_batch_fn
from src.datasets.wikiart_dataset import WikiArt
from src.datasets.fungi_dataset import FungiDataset
from src.datasets.coco_dataset import CocoDataset


class MetricDataloader(torch.utils.data.DataLoader):
  def __len__(self):
    return len(self.sampler)


def get_metric_dataloader(way,
                          shot,
                          batch_size,
                          transform,
                          split,
                          dataset: str = "imagenet",
                          dataset_kwargs={},
                          **kwargs):
  """Dataset is one of 'imagenet', 'wikiart'."""
  if (batch_size - way * shot) % way != 0:
    raise Exception(
      f'Batch size does not evenly divide into way*shot samples: ' +
      f'{(batch_size - way * shot) % way} remainder -- this needs to be 0.'
    )
  if kwargs.get('use_embedding_cache', False):
    embedding_cache_dir = kwargs.get('embedding_cache_dir')
    print('Using embeddings cached at', embedding_cache_dir)
    data = CachedEmbeddingDataset(embedding_cache_dir,
                                  split=split,
                                  **dataset_kwargs)
  else:
    # Otherwise, we load the data normally
    if dataset == "imagenet":
      data = EpisodicImageNet('../image_datasets/latest_imagenet', split=split, transform=transform)
    elif dataset == "wikiart-style":
      data = WikiArt(split=split, class_column="style", transform=transform)
    elif dataset == "wikiart-genre":
      data = WikiArt(split=split, class_column="genre", transform=transform)
    elif dataset == "wikiart-artist":
      data = WikiArt(split=split, class_column="artist", transform=transform)
    elif dataset == 'fungi':
      data = FungiDataset(split=split, transform=transform)
    elif dataset == 'coco':
      data = CocoDataset(split=split, transform=transform)
  num_workers = 50
  episodic_sampler = MetricSampler(len(data.classes),
                                   data.target_to_index,
                                   way=way,
                                   shot=shot,
                                   batch_size=batch_size)
  data_loader = MetricDataloader(data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 sampler=episodic_sampler,
                                 collate_fn=custom_collate_batch_fn)
  return data_loader


if __name__ == '__main__':
  transforms = Compose([
    CenterCrop(size=(224, 224)),
    ToTensor(),
    Normalize(mean=torch.tensor([0.4815, 0.4578, 0.4082]), std=torch.tensor([0.2686, 0.2613, 0.2758]))]
  )
  dataloader = get_metric_dataloader(5, 5, 525, transforms, 'train', 'wikiart-artist')
  for i,(f, l) in enumerate(dataloader):
    print(f.shape)
    print(i)
  print(len(dataloader))
