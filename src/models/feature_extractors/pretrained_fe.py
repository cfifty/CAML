import torch
import timm
import sys

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

import src.datasets.transforms as fe_transforms


def get_fe_metadata(args):
    """
    Returns a dict with some keys, including:
    'model': model obect
    'embed_dim': int
    'transform': transform
    'dtype': torch.dtype
    """
    fe_type = args.fe_type
    if args.fe_dtype == 'float16':
        dtype = torch.float16
    elif args.fe_dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        print('Defaulting to float32 dtype')
        dtype = torch.float32
    # Timm expects a format of timm:<model_name>:<feature_dim>
    if fe_type.startswith('timm:'):
        pieces = fe_type.split(':')
        assert len(pieces) == 3
        dim = int(pieces[-1])
        timm_model_name = pieces[1]
        feature_extractor = get_timm_model(timm_model_name, args.model, dtype=dtype)
        transforms = fe_transforms.get_timm_transform(feature_extractor)
    # Cache expects a format of cache:<fe_type>:<feature_dim>
    elif fe_type.startswith('cache:'):
        pieces = fe_type.split(':')
        # assert len(pieces) == 3
        dim = int(pieces[-1])
        feature_extractor = torch.nn.Identity()
        transforms = fe_transforms.get_empty_transform()
    else:
        raise Exception(f'Fe type: {fe_type} not recognized.')
    if not transforms:
        raise ValueError(f'Transform not implemented for fe type: {fe_type}')
    metadata = {
        'fe': feature_extractor,
        'fe_dim': dim,
        'train_transform': transforms['train_transform'],
        'test_transform': transforms['test_transform'],
        'dtype': dtype
    }
    return metadata


def get_timm_model(model_name, model_type, dtype=None):
    # model types that only rely on fixed pretrained backbones
    if model_type in ['CAML', 'MetaQDA', 'SNAIL'] or 'ICL' in model_type:
        if 'clip' in model_name:
            model = timm.create_model(model_name,
                                      pretrained=True,
                                      img_size=224,
                                      num_classes=0).eval()
        else:
            model = timm.create_model(model_name,
                                      pretrained=True,
                                      num_classes=0).eval()
    elif model_type.startswith('Proto:') or model_type == 'MetaOpt' or model_type == 'Proto':
        if 'clip' in model_name:
            model = timm.create_model(model_name,
                                      pretrained=True,
                                      img_size=224,
                                      num_classes=0).train()
        else:
            model = timm.create_model(model_name,
                                      pretrained=True,
                                      num_classes=0).train()
    else:
        raise Exception(
            f'Meta learning algorithm {model_type} is not recognized.')
    if dtype:
        model = model.to(dtype)
    print('Loaded pretrained timm model', model_name)
    return model
