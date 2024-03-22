import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.blocks.metaOptLinear import ClassificationHead

class MetaOptNet(nn.Module):
  def __init__(self,
               feature_extractor,
               fe_dim,
               fe_dtype,
               device=torch.device('cuda:0'),
               **kwargs):

    super().__init__()
    self.feature_extractor = feature_extractor
    self.fe_dim = fe_dim
    self.fe_dtype = fe_dtype
    self.device = device
    self.linear_model = ClassificationHead(base_learner='SVM-CS')

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
    # Features will be |shot| contiguous elements.
    features = self.get_feature_vector(inp)
    b, d = features.shape

    # Compute the centroids over the "shot" dimension -- aggregates all points belonging to the same class together.
    support = features[:way * shot].reshape(1, way * shot, d)
    query = features[way * shot:].reshape(1, -1, d)
    support_labels = labels.reshape(1, -1)

    logits = self.linear_model(query, support, support_labels, way, shot).squeeze()
    return F.log_softmax(logits, dim=1)

  def meta_test(self, inp, way, shot, query_shot):
    # Features will be |shot| contiguous elements.
    features = self.get_feature_vector(inp)
    b, d = features.shape

    # Compute the centroids over the "shot" dimension -- aggregates all points belonging to the same class together.
    support = features[:way * shot].reshape(1, way * shot, d)
    query = features[way * shot:].reshape(1, -1, d)
    support_labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device).reshape(1, -1)

    logits = self.linear_model(query, support, support_labels, way, shot).squeeze()
    _, max_index = torch.max(logits, 1)
    return max_index
