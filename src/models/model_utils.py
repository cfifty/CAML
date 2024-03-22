import sys

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.Proto import Proto
from src.models.GPICL import GPICL
from src.models.CAML import CAML
from src.models.SNAIL import SNAIL
from src.models.MetaOptNet import MetaOptNet
from src.models.MetaQDA import MetaQDA


def get_model_by_type(args, fe_metadata, device):
  if args.model == 'CAML':
    model = CAML(feature_extractor=fe_metadata['fe'],
                fe_dim=fe_metadata['fe_dim'],
                fe_dtype=fe_metadata['dtype'],
                train_fe=args.train_fe,
                encoder_size=args.encoder_size,
                dropout=args.dropout,
                label_elmes=args.label_elmes,
                device=device,
                set_transformer=args.set_transformer)
  elif args.model == 'SNAIL':
    model = SNAIL(feature_extractor=fe_metadata['fe'],
                  fe_dim=fe_metadata['fe_dim'],
                  fe_dtype=fe_metadata['dtype'],
                  train_fe=args.train_fe,
                  device=device)
  elif args.model == 'GPICL':
    model = GPICL(feature_extractor=fe_metadata['fe'],
                  fe_dim=fe_metadata['fe_dim'],
                  fe_dtype=fe_metadata['dtype'],
                  train_fe=args.train_fe,
                  encoder_size=args.encoder_size,
                  dropout=args.dropout,
                  label_elmes=args.label_elmes,
                  device=device)
  elif args.model == 'Proto':
    model = Proto(feature_extractor=fe_metadata['fe'],
                  fe_dim=fe_metadata['fe_dim'],
                  fe_dtype=fe_metadata['dtype'],
                  dropout=args.dropout,
                  device=device)
  elif args.model == 'MetaQDA':
    model = MetaQDA(feature_extractor=fe_metadata['fe'],
                    fe_dim=fe_metadata['fe_dim'],
                    fe_dtype=fe_metadata['dtype'],
                    device=device)
  elif args.model == 'MetaOpt':
    model = MetaOptNet(feature_extractor=fe_metadata['fe'],
                       fe_dim=fe_metadata['fe_dim'],
                       fe_dtype=fe_metadata['dtype'],
                       dropout=args.dropout,
                       device=device)
  else:
    raise Exception('model not recognized.')
  return model
