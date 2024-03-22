import sys
import os

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.CAML import CAML
from src.models.GPICL import GPICL
from src.models.SNAIL import SNAIL
from src.models.Proto import Proto
from src.models.MetaOptNet import MetaOptNet
from src.models.MetaQDA import MetaQDA

def get_model(args, fe_metadata, device):
  """Get the model and the model_path for meta-testing."""
  if args.model == 'CAML':
    if 'clip' in args.fe_type:
      model_path = '../caml_pretrained_models/CAML_CLIP/model.pth'
    elif 'laion' in args.fe_type:
      model_path = '../caml_pretrained_models/CAML_Laion2b/model.pth'
    elif 'resnet' in args.fe_type:
      model_path = '../caml_pretrained_models/CAML_ResNet34/model.pth'
    else:
      raise Exception(f'{args.fe_type} is not recognized.')
    model = CAML(feature_extractor=fe_metadata['fe'],
                fe_dim=fe_metadata['fe_dim'],
                fe_dtype=fe_metadata['dtype'],
                train_fe=False,
                encoder_size='large',
                device=device,
                label_elmes=True,
                dropout=args.dropout)
  elif args.model == 'SNAIL':
    model_path = '../SNAIL/model.pth'
    model = SNAIL(feature_extractor=fe_metadata['fe'],
                  fe_dim=fe_metadata['fe_dim'],
                  fe_dtype=fe_metadata['dtype'],
                  train_fe=False,
                  device=device)
  elif args.model == 'GPICL':
    model_path = '../caml_pretrained_models/GPICL/model.pth'
    model = GPICL(feature_extractor=fe_metadata['fe'],
                fe_dim=fe_metadata['fe_dim'],
                fe_dtype=fe_metadata['dtype'],
                train_fe=False,
                encoder_size='large',
                device=device,
                label_elmes=True,
                dropout=args.dropout)
  elif args.model == 'Proto':
    model_path = None  # This is correct: Proto performs best w/o finetuning pre-trained backbone.
    model = Proto(feature_extractor=fe_metadata['fe'],
                  fe_dim=fe_metadata['fe_dim'],
                  fe_dtype=fe_metadata['dtype'],
                  dropout=args.dropout,
                  device=device)
  elif args.model == 'MetaQDA':
    model_path = '../caml_pretrained_models/MetaQDA/model.pth'
    model = MetaQDA(feature_extractor=fe_metadata['fe'],
                    fe_dim=fe_metadata['fe_dim'],
                    fe_dtype=fe_metadata['dtype'],
                    device=device)
  elif args.model == 'MetaOpt':
    model_path = None  # This is correct: MetaOpt performs best w/o finetuning pre-trained backbone.
    model = MetaOptNet(feature_extractor=fe_metadata['fe'],
                       fe_dim=fe_metadata['fe_dim'],
                       fe_dtype=fe_metadata['dtype'],
                       dropout=args.dropout,
                       device=device)
  else:
    raise Exception(f'Model name {args.model} is not recognized')
  return model, model_path


def get_test_path(args, data_path):
  """Get the path to the meta-testing dataset."""
  if args.eval_dataset == 'meta_iNat':
    test_path = os.path.join(data_path, 'meta_iNat/test')
  elif args.eval_dataset == 'tiered_ImageNet':
    test_path = os.path.join(data_path, 'tiered-ImageNet_DeepEMD/test')
  elif args.eval_dataset == 'ChestX':
    test_path = os.path.join(data_path, 'ChestX/test')
  elif args.eval_dataset == 'CUB_fewshot':
    test_path = os.path.join(data_path, 'CUB_fewshot_raw/test')
  elif args.eval_dataset == 'Aircraft':
    test_path = os.path.join(data_path, 'Aircraft_fewshot/test')
  elif args.eval_dataset == 'tiered_meta_iNat':
    test_path = os.path.join(data_path, 'tiered_meta_iNat/test')
  elif args.eval_dataset == 'mini_ImageNet':
    # mini-ImageNet/test_pre contains the pre-resized 84x84 rgb images.
    test_path = os.path.join(data_path, 'mini-ImageNet/test_pre')
  elif args.eval_dataset == 'cifar':
    # cifar-fs/test contains the resized 32x32 rgb images.
    test_path = os.path.join(data_path, 'cifar-fs/test')
  elif args.eval_dataset == 'paintings':
    test_path = os.path.join(data_path, 'v2_paintings/test')
  elif args.eval_dataset == 'pascal_paintings':
    test_path = os.path.join(data_path, 'v2_pascal_paintings/test')
  elif args.eval_dataset == 'pascal':
    test_path = os.path.join(data_path, 'v2_pascal/test')
  else:
    raise Exception(f'eval dataset in classic_test: {args.eval_dataset} is not recognized.')
  return test_path

