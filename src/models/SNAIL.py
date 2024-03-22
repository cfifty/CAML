# Adapted from https://github.com/eambutu/snail-pytorch.
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.blocks.snail_blocks import AttentionBlock, TCBlock


class SNAIL(nn.Module):
  def __init__(self, feature_extractor, fe_dim, fe_dtype, train_fe, device=torch.device('cuda:0')):
    super().__init__()
    self.feature_extractor = feature_extractor
    self.fe_dim = fe_dim
    self.fe_dtype = fe_dtype
    self.train_fe = train_fe
    self.num_classes = 5
    self.device = device

    # Freeze weights in the Feature Extractor if train_fe is False.
    if not self.train_fe:
      for p in self.feature_extractor.parameters():
        p.requires_grad = False
    self.zeros_embedding = torch.zeros((1, 1, self.num_classes)).to(device)

    # SNAIL Model parameters.
    num_channels = 768 + self.num_classes
    N = 5
    K = 5
    num_filters = int(math.ceil(math.log(N * K + 1, 2)))

    self.attention1 = AttentionBlock(num_channels, 64, 32)
    num_channels += 32
    self.tc1 = TCBlock(num_channels, N * K + 1, 128)
    num_channels += num_filters * 128
    self.attention2 = AttentionBlock(num_channels, 256, 128)
    num_channels += 128
    self.tc2 = TCBlock(num_channels, N * K + 1, 128)
    num_channels += num_filters * 128
    self.attention3 = AttentionBlock(num_channels, 512, 256)
    num_channels += 256
    self.fc = nn.Linear(num_channels, N)
    self.N = N
    self.K = K

  def get_feature_vector(self, inp):
    batch_size = inp.size(0)
    origin_dtype = inp.dtype
    if origin_dtype != self.fe_dtype:
      inp = inp.to(self.fe_dtype)
    feature_map = self.feature_extractor(inp)
    if feature_map.dtype != origin_dtype:
      feature_map = feature_map.to(origin_dtype)
    feature_vector = feature_map.view(batch_size, self.fe_dim)
    return feature_vector

  def forward(self, inp, labels, way, shot):
    """For large-scale pre-training."""
    with torch.no_grad():
      features = self.get_feature_vector(inp)
    _, d = features.shape
    query = features[way * shot:].reshape(-1, 1, d)
    b, _, _ = query.shape

    # print(labels.shape)
    # print(labels)
    # labels = torch.flip(labels, (0,))
    # print(labels)
    # print(query.shape)
    # query = torch.flip(query, (0,))
    # print(query.shape)
    # raise

    # Repeat the support |B| times for each query example.
    support = features[:way * shot].reshape(1, way * shot, d).repeat(b, 1, 1)
    feature_sequences = torch.cat([support, query], dim=1)

    # Encode the labels: represent the query as the zeros vector.
    label_one_hot = F.one_hot(labels.unsqueeze(0), num_classes=self.num_classes).to(torch.float32)
    batched_label_embeddings = torch.cat([label_one_hot, self.zeros_embedding], dim=1).repeat(b, 1, 1)

    demonstrations = torch.cat([feature_sequences, batched_label_embeddings], dim=-1)

    demonstrations = self.attention1(demonstrations)
    demonstrations = self.tc1(demonstrations)
    demonstrations = self.attention2(demonstrations)
    demonstrations = self.tc2(demonstrations)
    demonstrations = self.attention3(demonstrations)
    logits = self.fc(demonstrations[:, -1, :])
    return logits


  def meta_test(self, inp, way, shot, query_shot):
    """For evaluating typical Meta-Learning Datasets."""
    feature_vector = self.get_feature_vector(inp)
    support_features = feature_vector[:way * shot]
    query_features = feature_vector[way * shot:]
    b, d = query_features.shape

    # Reshape query and support to a sequence.
    support = support_features.reshape(1, way * shot, d).repeat(b, 1, 1)
    query = query_features.reshape(-1, 1, d)
    labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device)

    labels = torch.flip(labels, (0,)) # Flip labels because SNAIL is used to [(4, 3, 2, 1, 0)] during training & not perm-equivariant.
    query = torch.flip(query, (0,))  # Flip query because

    feature_sequences = torch.cat([support, query], dim=1)

    # Encode the labels: represent the query as the zeros vector.
    label_one_hot = F.one_hot(labels.unsqueeze(0), num_classes=self.num_classes).to(torch.float32)
    batched_label_embeddings = torch.cat([label_one_hot, self.zeros_embedding], dim=1).repeat(b, 1, 1)

    # Input sequence is concatenation of features + labels.
    demonstrations = torch.cat([feature_sequences, batched_label_embeddings], dim=-1)

    # Pass through SNAIL architecture.
    demonstrations = self.attention1(demonstrations)
    demonstrations = self.tc1(demonstrations)
    demonstrations = self.attention2(demonstrations)
    demonstrations = self.tc2(demonstrations)
    demonstrations = self.attention3(demonstrations)
    logits = self.fc(demonstrations[:, -1, :])

    _, max_index = torch.max(logits, 1)
    return max_index
