import math
import sys
import torch

import numpy as np
import torch.nn as nn

from collections import OrderedDict
from functools import partial
from torchvision.models.vision_transformer import MLPBlock
from typing import Any, Callable

from torch.nn import functional as F

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


def get_encoder(size, image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  if size == 'tiny':
    return tiny_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'small':
    return small_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'base':
    return short_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'base':
    return base_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'large':
    return large_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'convnext':
    return convnext_32(image_dim, num_classes, device=device, **kwargs)
  elif size == 'laion':
    return laion_2b(image_dim, num_classes, device=device, **kwargs)
  elif size == 'resnet34':
    return resnet34(image_dim, num_classes, device=device, **kwargs)
  elif size == 'huge':
    return huge_32(image_dim, num_classes, device=device, **kwargs)


def tiny_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=4,
                            num_heads=8,
                            hidden_dim=1024,
                            mlp_dim=1024,
                            device=device,
                            **kwargs,
                            )


def small_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=8,
                            num_heads=8,
                            hidden_dim=1024,
                            mlp_dim=1024,
                            device=device,
                            **kwargs,
                            )


def short_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=4,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            device=device,
                            **kwargs,
                            )


def base_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            device=device,
                            **kwargs,
                            )


def large_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=24,
                            num_heads=16,
                            hidden_dim=1024,
                            mlp_dim=4096,
                            device=device,
                            **kwargs,
                            )


def resnet34(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=24,
                            num_heads=16,
                            hidden_dim=768,
                            mlp_dim=4096,
                            device=device,
                            **kwargs,
                            )


def convnext_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=24,
                            num_heads=16,
                            hidden_dim=1280,
                            mlp_dim=4096,
                            device=device,
                            **kwargs,
                            )


def laion_2b(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=24,
                            num_heads=16,
                            hidden_dim=1536,
                            mlp_dim=4096,
                            device=device,
                            **kwargs,
                            )


def huge_32(image_dim, num_classes, *, device=torch.device('cuda:0'), **kwargs: Any):
  return TransformerEncoder(image_dim=image_dim, num_classes=num_classes,
                            num_layers=32,
                            num_heads=16,
                            hidden_dim=1280,
                            mlp_dim=5120,
                            device=device,
                            **kwargs,
                            )


def get_elmes(p, C, cuda):
  ones = torch.ones((C, 1), dtype=torch.float32)
  M_star = torch.sqrt(torch.tensor(C / (C - 1))) * (
          torch.eye(C) - 1 / C * torch.matmul(ones, torch.transpose(ones, 0, 1)))
  np.random.seed(50)
  U = np.random.random(size=(p, C))
  U = torch.tensor(np.linalg.qr(U)[0][:, :C]).to(torch.float32)
  return (U @ M_star).T.to(cuda)


class TransformerEncoder(nn.Module):
  def __init__(self,
               image_dim,
               num_classes,
               num_layers: int,
               num_heads: int,
               hidden_dim: int,
               mlp_dim: int,
               device: torch.device,
               dropout: float = 0.0,
               attention_dropout: float = 0.0,
               norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
               label_elmes=True,
               GPICL=False,
               set_transformer=False):
    super().__init__()
    self.image_dim = image_dim
    self.hidden_dim = hidden_dim
    self.mlp_dim = mlp_dim
    self.attention_dropout = attention_dropout
    self.dropout = dropout
    self.norm_layer = norm_layer
    if set_transformer:
      encoder = Encoder(
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
        set_transformer=True,
      )
      self.encoder = encoder
    else:
      self.encoder = Encoder(
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        norm_layer,
      )
    self.device = device
    self.num_classes = num_classes

    self.feature_proj = torch.nn.Identity()
    self.unk_emb = torch.nn.Parameter(torch.zeros(1, 1, 256))
    self.elmes_scale = torch.nn.Parameter(torch.ones(1))
    # Change to a non-trainable Parameter so this loads from model weights (in case random changes).
    if label_elmes:
      self.label_elmes = torch.nn.Parameter(get_elmes(256, num_classes, device), requires_grad=False)
    else:
      self.label_elmes = torch.nn.Parameter(torch.empty(num_classes, 256, device=device), requires_grad=True)
      torch.nn.init.kaiming_uniform_(self.label_elmes, a=math.sqrt(5))

    self.output_proj = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=False)

    if GPICL:
      self.zeros_embedding = torch.zeros((1, 1, self.num_classes)).to(device)
      self.positional_embedding = nn.Parameter(torch.zeros(1, 26, hidden_dim)).to(device)  # Hardcode for rebuttal.
      self.demonstration_proj = nn.Linear(768 + 5, 1024).to(
        device)  # Hardcode for clip: 768 clip embeddings + 5 one hot-label encodings.

  def forward(self, features, labels):
    features = self.feature_proj(features)
    b, _, _ = features.shape

    s = labels.shape
    label_one_hot = F.one_hot(labels.unsqueeze(0), num_classes=self.num_classes).to(torch.float32)
    label_embeddings = label_one_hot @ (self.elmes_scale * self.label_elmes)
    batched_label_embeddings = torch.cat([self.unk_emb, label_embeddings], dim=1).repeat(b, 1, 1)
    demonstrations = torch.cat([features, batched_label_embeddings], dim=-1)

    seq = self.encoder.forward(demonstrations)
    return seq

  def forward_imagenet(self, features, labels):
    # Assumes the first index in the sequence is the query.
    b, s, d = features.shape
    features = self.feature_proj(features)
    label_embeddings = (F.one_hot(labels.reshape(b * s, -1), num_classes=self.num_classes).to(
      torch.float32) @ (self.elmes_scale * self.label_elmes)).reshape(b, s, -1)
    label_embeddings[:, 0, :] = self.unk_emb
    demonstrations = torch.cat([features, label_embeddings], dim=-1)

    seq = self.encoder.forward(demonstrations)
    query = seq[:, 0, :]
    return self.output_proj(query)

  def forward_imagenet_v2(self, features, labels, way, shot):
    # Assumes the labels has length = len(features) -1 => need to cat the unk emb to the labels.
    seq = self.forward(features, labels)
    query = seq[:, 0, :]
    return self.output_proj(query)

  def forward_gpicl(self, features, labels, way, shot):
    features = self.feature_proj(features)
    b, _, _ = features.shape

    label_one_hot = F.one_hot(labels.unsqueeze(0), num_classes=self.num_classes).to(torch.float32)
    batched_label_embeddings = torch.cat([self.zeros_embedding, label_one_hot], dim=1).repeat(b, 1, 1)
    demonstrations = torch.cat([features, batched_label_embeddings], dim=-1)

    # GPICL relies on positional embeddings to classify the point.
    demonstrations = self.demonstration_proj(demonstrations) + self.positional_embedding[:, :demonstrations.shape[1], :]

    seq = self.encoder.forward(demonstrations)
    query = seq[:, -1, :]
    return self.output_proj(query)

  def forward_gpicl_with_dataloader(self, features, labels):
    # Assumes the **last** index in the sequence is the query.
    b, s, d = features.shape
    features = self.feature_proj(features)
    labels = labels[:, :-1]  # Remove the label for the query: labels now has sequence length (s-1).
    # Concatenate the zeros vector with the sequence in the first position.
    label_embeddings = torch.cat([self.zeros_embedding,
                                  F.one_hot(labels.reshape(b * s, -1), num_classes=self.num_classes).to(
                                    torch.float32)]).reshape(b, s, -1)
    demonstrations = torch.cat([features, label_embeddings], dim=-1)
    demonstrations = self.demonstration_proj(demonstrations) + self.positional_embedding[:, :demonstrations.shape[1], :]

    seq = self.encoder.forward(demonstrations)
    query = seq[:, -1, :]
    return self.output_proj(query)


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(
          self,
          num_layers: int,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
          set_transformer=False,
  ):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for i in range(num_layers):
      if set_transformer:
        # Set transformer is heavily overparameterized: let's drop # layers from 24 => 12 for large model.
        if i % 6 == 0:
          layers[f"encoder_layer_{i}"] = SetEncoderBlock(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
          )
      else:
        layers[f"encoder_layer_{i}"] = EncoderBlock(
          num_heads,
          hidden_dim,
          mlp_dim,
          dropout,
          attention_dropout,
          norm_layer,
        )
    self.layers = nn.Sequential(layers)
    self.ln = norm_layer(hidden_dim)

  def forward(self, x: torch.Tensor):
    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    return self.ln(self.layers(self.dropout(x)))


class EncoderBlock(nn.Module):
  """Transformer encoder block."""

  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__()
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
    self.dropout = nn.Dropout(dropout)

    # MLP block
    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y


class SetEncoderBlock(EncoderBlock):
  def __init__(
          self,
          num_heads: int,
          hidden_dim: int,
          mlp_dim: int,
          dropout: float,
          attention_dropout: float,
          norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
  ):
    super().__init__(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
    self.self_attention_set = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
    self.ln_1_set = norm_layer(hidden_dim)
    self.dropout_set = nn.Dropout(dropout)

    # MLP block
    self.ln_2_set = norm_layer(hidden_dim)
    self.mlp_set = MLPBlock(hidden_dim, mlp_dim, dropout)

    self.I = nn.Parameter(torch.Tensor(1, 16, hidden_dim), requires_grad=True)  # Hardcode to 16.
    nn.init.xavier_uniform_(self.I)

  def forward(self, input: torch.Tensor):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

    # Set-level attention.
    # x = self.ln_1_set(input)
    x = self.ln_1(input)
    x_set, _ = self.self_attention_set(query=self.I.repeat((x.shape[0], 1, 1)), key=x, value=x, need_weights=False)
    x_set = self.dropout_set(x_set)
    x_set = x_set + self.I

    y_set = self.ln_2_set(x_set)
    y_set = self.mlp_set(y_set)
    y_set = x_set + y_set

    # Normal attention.
    x, _ = self.self_attention(query=x, key=y_set, value=y_set, need_weights=False)
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y
