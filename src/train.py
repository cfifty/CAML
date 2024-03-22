import sys
import torch

from torch.nn import CrossEntropyLoss, NLLLoss
from functools import partial
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.datasets.dataloaders import get_metric_dataloader
from src.datasets.dataset_utils import get_dataloaders

from src.train_utils.eval_loop import metric_eval_fn
from src.train_utils.train_loop import metric_train_fn
from src.train_utils.trainer import train_parser, Train_Manager

from src.models.model_utils import get_model_by_type
from src.models.feature_extractors.pretrained_fe import get_fe_metadata
"""
We evaluate CAML with CLIP, ResNet34, and Laion-2b feature extractors: 
  --fe_type cache:timm:vit_base_patch16_clip_224.openai:768 
  --fe_type cache:timm:vit_huge_patch14_clip_224.laion2b:1280
  --fe_type cache:timm:resnet34:512
  
The MetaQDA library is written in a single GPU paradigm, so run 
* export CUDA_VISIBLE_DEVICES=0
and then set
--gpu 0
when running this file.

The following is an example script to kick off a training run. To evaluate a model on a downstream benchmark, see 
src/evaluation/test.py

Example 1: Train CAML with a CLIP image encoder and ELMES label encoder. 
python src/train.py \
     --opt adam \
     --lr 1e-5 \
     --epoch 100 \
     --val_epoch 1 \
     --batch_sizes 525 \
     --detailed_name \
     --encoder_size large \
     --dropout 0.0  \
     --fe_type cache:timm:vit_base_patch16_clip_224.openai:768 \
     --schedule custom_cosine \
     --fe_dtype float32 \
     --model CAML \
     --label_elmes \
     --save_dir test_CAML_repo \
     --gpu 2
     
Example 2: Train CAML with a Laion-2b image encoder and ELMES label encoder.
python src/train.py \
     --opt adam \
     --lr 1e-5 \
     --epoch 100 \
     --val_epoch 1 \
     --batch_sizes 525 \
     --detailed_name \
     --encoder_size laion \
     --dropout 0.0  \
     --fe_type cache:timm:vit_huge_patch14_clip_224.laion2b:1280 \
     --schedule custom_cosine \
     --fe_dtype float32 \
     --model CAML \
     --label_elmes \
     --save_dir test_CAML_repo \
     --gpu 2
     
Example 3: Train MetaQDA with a CLIP image encoder.
  --> See Line 7 of MetaQDA.py to set the DEVICE variable.
python src/train.py \
     --opt adam \
     --lr 1e-5 \
     --epoch 100 \
     --val_epoch 1 \
     --batch_sizes 525 \
     --detailed_name \
     --encoder_size large \
     --dropout 0.0  \
     --fe_type cache:timm:vit_base_patch16_clip_224.openai:768 \
     --schedule custom_cosine \
     --fe_dtype float32 \
     --model MetaQDA \
     --label_elmes \
     --save_dir test_CAML_repo \
     --gpu 2
"""

args = train_parser()
assert len(args.batch_sizes) == 1  # Batch sizes now = 1.

# Cache expects a format of cache:<fe_type>:<feature_dim>
use_embedding_cache = args.fe_type.startswith('cache:')
fe_subdir = ''
if use_embedding_cache:
    fe_subdir = ':'.join(args.fe_type.split(':')[1:])

# Figure out what feature extractor we're using and get associated metadata.
fe_metadata = get_fe_metadata(args)

# Load the dataset.
train_transforms = fe_metadata['train_transform']
test_transforms = fe_metadata['test_transform']
device = torch.device(f'cuda:{args.gpu}')

# Load the model.
model = get_model_by_type(args, fe_metadata, device)
model.to(device)

# Train on 5-way-5-shot and 5-way-1-shot [in that order].
dataset_spec = [('imagenet', 5, 5), ('imagenet', 5, 1),
                ('wikiart-style', 5, 5), ('wikiart-style', 5, 1),
                ('fungi', 5, 5), ('fungi', 5, 1), ('wikiart-genre', 5, 5),
                ('wikiart-genre', 5, 1), ('coco', 5, 5), ('coco', 5, 1),
                ('wikiart-artist', 5, 5), ('wikiart-artist', 5, 1)
                ]
train_dataloaders = get_dataloaders(datasets=dataset_spec,
                                    split='train',
                                    dataloader_fn=get_metric_dataloader,
                                    batch_size=args.batch_sizes[0],
                                    transforms=train_transforms,
                                    use_embedding_cache=use_embedding_cache,
                                    fe_subdir=fe_subdir)
valid_dataloaders = get_dataloaders(datasets=dataset_spec,
                                    split='val',
                                    dataloader_fn=get_metric_dataloader,
                                    batch_size=args.batch_sizes[0],
                                    transforms=test_transforms,
                                    use_embedding_cache=use_embedding_cache,
                                    fe_subdir=fe_subdir)

# Wrap train and eval loops in a callable.
num_evals = 2
# Order is important! MetaOpt needs to come first due to MetaOptICL taking NLLLoss().
if 'MetaOpt' in args.model:
    train_func = partial(metric_train_fn,
                         train_loaders=train_dataloaders,
                         criterion=NLLLoss().to(device))
    valid_func = partial(metric_eval_fn,
                         eval_loaders=valid_dataloaders,
                         num_loops=num_evals,
                         criterion=NLLLoss().to(device))
elif args.model in ['MetaQDA', 'Proto', 'SNAIL'] or 'CAML' in args.model:
    train_func = partial(metric_train_fn,
                         train_loaders=train_dataloaders,
                         criterion=CrossEntropyLoss().to(device))
    valid_func = partial(metric_eval_fn,
                         eval_loaders=valid_dataloaders,
                         num_loops=num_evals,
                         criterion=CrossEntropyLoss().to(device))

else:
    raise Exception(f'Model "{args.model}" is not recognized')

tm = Train_Manager(args,
                   train_func=train_func,
                   valid_func=valid_func,
                   dataset_spec=dataset_spec)
tm.train(model)
