import torch


def metric_train_fn(train_loaders, model, criterion, optimizer, scheduler, iter_counter, **kwargs):
  model.train()

  losses, accs = [], []
  for train_loader in train_loaders:
    way = train_loader.sampler.way
    shot = train_loader.sampler.shot
    avg_loss = avg_acc = 0
    for i, (inp, labels) in enumerate(train_loader):
      inp = inp.to(model.device)

      # Consistently translate arbitrary integer labels => labels \in [0, 4].
      unique_labels = torch.unique(labels, sorted=False)
      bool_vec = (labels == unique_labels.unsqueeze(1))
      labels = torch.max(bool_vec, dim=0)[1].to(torch.int64).to(inp.device)
      support_labels = labels[:way*shot]
      # Perhaps a support set point was marked as a duplicate by approximating image equality with the norm.
      if torch.unique(support_labels).shape[0] != 5:
        print(f'Malformed batch with only {torch.unique(support_labels).shape[0]} unique support examples. Skipping.')
        continue
      query_labels = labels[way*shot:]
      query_labels = torch.flip(query_labels, dims=[0])

      logits = model(inp, support_labels, way, shot)
      loss = criterion(logits, query_labels)

      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      loss_value = loss.item()
      _, max_index = torch.max(logits, 1)
      # if i % 50 == 0:
      #   print(max_index)
      #   print(query_labels)
      acc = 100 * torch.sum(torch.eq(max_index, query_labels)).item() / query_labels.shape[0]
      avg_acc += acc
      avg_loss += loss_value

      if i % 50 == 0:
        print(f'loss at step {i}: {loss:.3f}')
        print(f'accuracy at step {i}: {acc:.3f}')

    avg_acc = avg_acc / (i + 1)
    avg_loss = avg_loss / (i + 1)

    print(f'average train loss: {avg_loss:3f}')
    accs.append(avg_acc)
    losses.append(avg_loss)
  return iter_counter, losses, accs

