import argparse
import logging
import os
import sys
import torch

from tqdm import tqdm

import numpy as np
import torch.optim as optim
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.train_utils.train_utils import WarmupCosineSchedule, CustomWarmupCosineSchedule


def get_logger(filename):
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                datefmt='%m/%d %I:%M:%S')
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  fh = logging.FileHandler(filename, "w")
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  sh = logging.StreamHandler()
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  return logger


def train_parser():
  parser = argparse.ArgumentParser()

  ## general hyper-parameters
  parser.add_argument('--opt', help='optimizer', choices=['adam', 'sgd'])
  parser.add_argument('--schedule',
                      help='the learning rate scheduler',
                      choices=['cosine', 'constant', 'custom_cosine'])
  parser.add_argument('--model', help='The model type.')
  parser.add_argument('--label_elmes', help='Whether to use a ELMES as the label encoder or learn these vectors.',
                      action='store_true')
  parser.add_argument('--encoder_size',
                      help='Size of the transformer encoder')
  parser.add_argument('--lr', help='initial learning rate', type=float)
  parser.add_argument('--epoch', help='number of epochs to train', type=int)
  parser.add_argument('--weight_decay',
                      help='weight decay for optimizer',
                      type=float,
                      default=0.)
  parser.add_argument('--gpu', help='gpu device', type=int, default=0)
  parser.add_argument('--seed', help='random seed', type=int, default=42)
  parser.add_argument('--val_epoch',
                      help='number of epochs before eval on val',
                      type=int,
                      default=20)
  parser.add_argument(
    '--batch_sizes',
    help=
    'batch size used during pre-training. Highest way*shot listed first.',
    type=int,
    nargs='+',
    default=[130, 510])
  parser.add_argument(
    '--train_fe',
    help=
    'whether to update the feature encoder weights during meta-training',
    action='store_true')
  parser.add_argument('--fe_type',
                      help='which feature extractor to use',
                      type=str,
                      default='')
  parser.add_argument('--save_dir',
                      help='Where to save model files.',
                      type=str,
                      default='ImageICL_outputs')
  parser.add_argument(
    '--no_val',
    help="don't use validation set, just save model at final timestep",
    action='store_true')
  parser.add_argument('--detailed_name',
                      help='whether include training details in the name',
                      action='store_true')
  parser.add_argument('--dropout',
                      help='weight decay for optimizer',
                      type=float,
                      default=0.0)
  parser.add_argument('--fe_dtype',
                      help='dtype to use for the feature encoder',
                      type=str,
                      default='float32')
  parser.add_argument('--GPICL',
                      help='Set this argument if evaluating GPICL or training the model.',
                      action='store_true')
  parser.add_argument('--set_transformer',
                      help='Whether to use a set_transformer engine or not.',
                      action='store_true')
  parser.add_argument(
    '--eval_dataset',
    help='Optional argument specifying which dataset to evaluate.',
    type=str,
    default='')
  parser.add_argument(
    '--image_embedding_cache_dir',
    type=str,
    default=None,
    help='If set, the dir to read/write image embeddings from')
  parser.add_argument("--dataset",
                      help="what dataset to compute embeddings for",
                      type=str)

  args = parser.parse_args()

  return args


def get_opt(model, args):
  if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
  elif args.opt == 'sgd':
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
  else:
    raise Exception(f'optimizer {args.opt} not recognized.')

  if args.schedule == 'cosine':
    if args.batch_sizes[-1] == 510:
      warmup_steps = 7200  # warmup over 3 epochs
      t_total = 120000  # Decay over the 50 epochs.
    elif args.batch_sizes[-1] == 1200:
      warmup_steps = 3000  # warmup over 3 epochs.
      t_total = 50000  # Decay over the 50 epochs.
    else:
      warmup_steps = 1500
      t_total = 60000
    scheduler = WarmupCosineSchedule(optimizer,
                                     warmup_steps=warmup_steps,
                                     t_total=t_total)
  elif args.schedule == 'custom_cosine':
    warmup_steps = 9600
    t_total = 360000
    scheduler = CustomWarmupCosineSchedule(optimizer,
                                           warmup_steps=warmup_steps,
                                           t_total=t_total,
                                           final_lr=args.lr / 10)
  elif args.schedule == 'constant':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,
                                                    factor=1.0,
                                                    total_iters=0,
                                                    last_epoch=-1)
  else:
    raise Exception(f'Scheduler {args.schedule} is not recognized.')

  return optimizer, scheduler


class Train_Manager:

  def __init__(self, args, train_func, valid_func, dataset_spec):

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.set_device(args.gpu)
    name = f'{args.fe_type}'

    if args.detailed_name:
      suffix = '%s-%s-encoder_%s-schedule_%s-lr_%.0e-epoch_%d-batch_size%s-dropout%.0e' % (
        args.model, args.opt, args.encoder_size, args.schedule,
        args.lr, args.epoch, str(args.batch_sizes), args.dropout)
      name = "%s-%s" % (name, suffix)

    outer_dir = args.save_dir
    # outer_dir = 'outputs'
    os.makedirs(f'../{outer_dir}/{name}', exist_ok=True)
    self.logger = get_logger(f'../{outer_dir}/{name}/train.log')
    self.save_path = f'../{outer_dir}/{name}/model.pth'
    self.dataset_spec = dataset_spec

    self.logger.info('display all the hyper-parameters in args:')
    for arg in vars(args):
      value = getattr(args, arg)
      if value is not None:
        self.logger.info('%s: %s' % (str(arg), str(value)))
    self.logger.info('------------------------')
    self.args = args
    self.train_func = train_func
    self.valid_func = valid_func

  def train(self, model):

    args = self.args
    train_func = self.train_func
    valid_func = self.valid_func
    save_path = self.save_path
    logger = self.logger

    optimizer, scheduler = get_opt(model, args)

    best_val_acc = 0
    best_epoch = 0
    iter_counter = 0

    total_epoch = args.epoch
    logger.info("start training!")

    for e in tqdm(range(total_epoch)):
      # Compute training stats.
      iter_counter, train_losses, train_accs = train_func(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        iter_counter=iter_counter)
      logger.info("")
      logger.info("epoch %d/%d, iter %d:" % (e + 1, total_epoch, iter_counter))
      train_acc_str = ' '.join([
        f'dataset {ds} @{w}-{s} -> acc_{a:.3f}' for ((ds, w, s), a) in zip(self.dataset_spec, train_accs)
      ])
      train_loss_str = ' '.join([
        f'dataset {ds} @{w}-{s} -> loss_{l:.3f}' for ((ds, w, s), l) in zip(self.dataset_spec, train_losses)
      ])
      logger.info(f'train_acc: {train_acc_str}')
      logger.info(f'train_losses: {train_loss_str}')

      # Compute valid stats.
      with torch.no_grad():
        val_losses, val_accs = valid_func(model=model)
        val_acc_str = ' '.join([
          f'dataset {ds} @{w}-{s} -> acc_{a:.3f}' for ((ds, w, s), a) in zip(self.dataset_spec, val_accs)
        ])
        val_loss_str = ' '.join([
          f'dataset {ds} @{w}-{s} -> loss_{l:.3f}' for ((ds, w, s), l) in zip(self.dataset_spec, val_losses)
        ])
        logger.info(f'val_acc: {val_acc_str}')
        logger.info(f'val_losses: {val_loss_str}')

      # Save the best model by validation accuracy.
      if np.mean(val_accs) > best_val_acc:
        best_val_acc = np.mean(val_accs)
        best_epoch = e + 1
        torch.save(model.state_dict(), save_path)
        logger.info(f'Best epoch: {best_epoch}')

    logger.info('training finished!')
    logger.info('------------------------')
    logger.info(('the best epoch is %d/%d') % (best_epoch, total_epoch))
    logger.info(('the best val acc is %.3f') % (best_val_acc))
