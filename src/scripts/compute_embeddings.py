import os
import sys
import torch
import torchvision
import tqdm
import numpy as np

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.datasets.wikiart_dataset import WikiArt
from src.datasets.fungi_dataset import FungiDataset
from src.datasets.coco_dataset import CocoDataset
from src.train_utils.trainer import train_parser
from src.models.feature_extractors.pretrained_fe import get_fe_metadata

"""
See compute_embeddings.sh


python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type timm:vit_base_patch16_clip_224.openai:768 \
     --batch_size 1024 \
     --gpu 1 \
     --model ICL \
     --image_embedding_cache_dir ../latest_imagenet/cached_embeddings/ \
     --dataset wikiart-style
"""

args = train_parser()


# Figure out what feature extractor we're using and get associated metadata.
fe_metadata = get_fe_metadata(args)
transforms = fe_metadata['test_transform']
fe_model = fe_metadata['fe']
device = torch.device(f'cuda:{args.gpu}')
fe_model = fe_model.to(device)
batch_size = args.batch_sizes[0]


def _get_dataloader(dataset: str, split: str, batch_size, transforms):
  if dataset == 'imagenet':
    data = torchvision.datasets.ImageNet('../image_datasets/latest_imagenet', split=split, transform=transforms)
  elif dataset == 'wikiart-style':
    data = WikiArt(split=split, class_column='style', transform=transforms)
  elif dataset == 'wikiart-artist':
    data = WikiArt(split=split, class_column='artist', transform=transforms)
  elif dataset == 'wikiart-genre':
    data = WikiArt(split=split, class_column='genre', transform=transforms)
  elif dataset == 'fungi':
    data = FungiDataset(split=split, transform=transforms)
  elif dataset == 'coco':
    data = CocoDataset(split=split, transform=transforms)
  dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=12)
  return dataloader


for split in ['train', 'val']:
  # Load the dataset.
  dataloader = _get_dataloader(args.dataset, split, batch_size, transforms)

  cache_dir = os.path.join(args.image_embedding_cache_dir, args.fe_type, split)
  print('Writing embeddings to cache dir', cache_dir)

  # For computing average embeddings
  running_average = None
  examples_seen = 0

  for batch_idx, (images, labels) in tqdm.tqdm(enumerate(dataloader)):
    images = images.to(device)
    with torch.no_grad():
      embeddings = fe_model(images)

      # Update the average
      b = embeddings.shape[0]
      batch_mean = torch.mean(embeddings, dim=0)
      if running_average is None:
        running_average = batch_mean
      else:
        running_average = running_average * (
                examples_seen /
                (examples_seen + b)) + batch_mean * (b / examples_seen)
      examples_seen += b

      embeddings = embeddings.to('cpu').numpy()

      # Now write out the embeddings for each image individually
      for i in range(embeddings.shape[0]):
        embedding = embeddings[i, :].reshape(-1)
        cls = labels[i].reshape(-1).item()
        cls_dir = os.path.join(cache_dir, str(cls))
        if not os.path.exists(cls_dir):
          os.makedirs(cls_dir)
        example_idx = batch_idx * batch_size + i
        filename = os.path.join(cls_dir, f'{example_idx}.npy')
        np.save(filename, embedding)

  # Now write out the average
  running_average = running_average.to('cpu').numpy()
  average_filename = os.path.join(cache_dir, f'split_average.npy')
  np.save(average_filename, running_average)
