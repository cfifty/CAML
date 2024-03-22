import sys
import torch
import torch.nn as nn

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.models.TransformerEncoder import get_encoder


class GPICL(nn.Module):

  def __init__(self,
               feature_extractor,
               fe_dim,
               fe_dtype,
               train_fe,
               encoder_size,
               device=torch.device('cuda:0'),
               label_elmes=True,
               **kwargs):

    super().__init__()
    self.feature_extractor = feature_extractor
    self.fe_dim = fe_dim
    self.fe_dtype = fe_dtype
    self.train_fe = train_fe
    # Freeze weights in the Feature Extractor if train_fe is False.
    if not self.train_fe:
      for p in self.feature_extractor.parameters():
        p.requires_grad = False

    self.encoder_size = encoder_size
    self.device = device

    # kwargs sets dropout in transformer.
    self.transformer_encoder = get_encoder(encoder_size, image_dim=self.fe_dim, num_classes=5, device=device,
                                           label_elmes=label_elmes, GPICL=True, **kwargs)

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

    # Repeat the support |B| times for each query example.
    support = features[:way * shot].reshape(1, way * shot, d).repeat(b, 1, 1)

    feature_sequences = torch.cat([support, query], dim=1)
    logits = self.transformer_encoder.forward_gpicl(feature_sequences, labels, way, shot)
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
    query = torch.flip(query, (0,))
    feature_sequences = torch.cat([support, query], dim=1)

    labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device)
    labels = torch.flip(labels, (0,))
    logits = self.transformer_encoder.forward_gpicl(feature_sequences, labels, way, shot)
    _, max_index = torch.max(logits, 1)
    return max_index
