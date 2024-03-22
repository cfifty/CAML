def get_dataloaders(datasets, split, dataloader_fn, batch_size, transforms, use_embedding_cache, fe_subdir):
  rtn = []
  image_dataset_path = '../caml_train_datasets'
  for (dataset, way, shot) in datasets:
    if dataset == 'imagenet':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="imagenet",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/latest_imagenet/cached_embeddings/{fe_subdir}'))
    elif dataset == 'fungi':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="fungi",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/fungi/cached_embeddings/{fe_subdir}'))
    elif dataset == 'coco':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="coco",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/mscoco/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-style':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-style",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_style/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-genre':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-genre",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_genre/cached_embeddings/{fe_subdir}'))
    elif dataset == 'wikiart-artist':
      rtn.append(dataloader_fn(way=way,
                               shot=shot,
                               split=split,
                               dataset="wikiart-artist",
                               batch_size=batch_size,
                               transform=transforms,
                               use_embedding_cache=use_embedding_cache,
                               embedding_cache_dir=f'{image_dataset_path}/wikiart_artist/cached_embeddings/{fe_subdir}'))
  return rtn
