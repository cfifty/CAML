import numpy as np
import sys
import torch

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))



def metric_eval_fn(eval_loaders, model, criterion, num_loops=1):
  model.eval()
  accs, losses = [], []
  for eval_loader in eval_loaders:
    way = eval_loader.sampler.way
    shot = eval_loader.sampler.shot
    per_loop_accs, per_loop_losses = [], []
    for k in range(num_loops):
      for i, (inp, labels) in enumerate(eval_loader):
        inp = inp.to(model.device)

        # Consistently translate arbitrary integer labels => labels \in [0, 4].
        unique_labels = torch.unique(labels, sorted=False)
        bool_vec = (labels == unique_labels.unsqueeze(1))
        labels = torch.max(bool_vec, dim=0)[1].to(torch.int64).to(inp.device)
        support_labels = labels[:way * shot]
        # Perhaps a support set point was marked as a duplicate by approximating image equality with the norm.
        if torch.unique(support_labels).shape[0] != 5:
          print(f'Malformed batch with only {torch.unique(support_labels).shape[0]} unique support examples. Skipping.')
          continue
        query_labels = labels[way * shot:]
        query_labels = torch.flip(query_labels, dims=[0])

        logits = model(inp, support_labels, way, shot)
        loss = criterion(logits, query_labels)

        _, max_index = torch.max(logits, 1)
        acc = 100 * torch.sum(torch.eq(max_index, query_labels)).item() / query_labels.shape[0]
        per_loop_accs.append(acc)
        per_loop_losses.append(loss.item())
    accs.append(np.mean(per_loop_accs))
    losses.append(np.mean(per_loop_losses))
  return losses, accs
