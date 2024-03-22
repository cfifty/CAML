import torch

import numpy as np
import torch.nn as nn

torch.set_default_tensor_type(torch.FloatTensor)
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


class MetaQDA_MAP(nn.Module):
  def __init__(self, x_dim, reg_param=0.5):
    super(MetaQDA_MAP, self).__init__()
    self.x_dim = x_dim
    self.reg_param = reg_param
    self.m = torch.nn.Parameter(torch.zeros(1, self.x_dim))
    self.kappa = torch.nn.Parameter(torch.tensor(0.1))
    self.nu = torch.nn.Parameter(torch.tensor(self.x_dim, dtype=torch.float32))
    self.triu_diag = torch.nn.Parameter(torch.ones(self.x_dim))
    self.triu_lower = torch.nn.Parameter(torch.eye(self.x_dim))
    self.triu_mask = torch.triu(torch.ones(self.x_dim, self.x_dim), diagonal=1).t().to(DEVICE)

  def fit_image_label(self, X, y):
    self.classes = np.unique(y.cpu())
    self.mu = [None for i in self.classes]
    self.sigma_inv = [None for i in self.classes]
    self.lower_triu = torch.diag(torch.abs(self.triu_diag)) + self.triu_lower * self.triu_mask

    kappa_ = torch.abs(self.kappa) + 1e-6
    nu_ = torch.clamp(self.nu, min=self.x_dim - 1 + 1e-6)
    for j in self.classes:
      X_j = X[y == j]
      N_j = X_j.shape[0]
      self.mu[j] = kappa_.div(kappa_ + N_j) * self.m + torch.div(N_j, kappa_ + N_j) * torch.mean(X_j, dim=0,
                                                                                                 keepdim=True)
      sigma_part = self_outer(self.lower_triu) + sum_outer(X_j) + kappa_ * self_outer(self.m) - (
                kappa_ + N_j) * self_outer(self.mu[j])
      sigma_j = torch.div(sigma_part, nu_ + N_j + self.x_dim + 2)
      self.sigma_inv[j] = self.regularize(torch.inverse(sigma_j))

  def predict(self, X):
    predicts_matrix = []
    for i in range(X.shape[0]):
      neg_distrances = []
      for mu, sigma_inv in zip(self.mu, self.sigma_inv):
        diff = X[i, :] - mu
        gaussian_dist = torch.mm(torch.mm(diff, sigma_inv), diff.t())
        neg_distrances.append(-1 * gaussian_dist)

      predicts_matrix.append(torch.cat(neg_distrances, dim=1))
    predicts_matrix = torch.cat(predicts_matrix, dim=0)
    return predicts_matrix

  def regularize(self, sigma):
    return (1 - self.reg_param) * sigma + self.reg_param * torch.eye(self.x_dim).to(DEVICE)


class MetaQDA_FB(nn.Module):
  def __init__(self, x_dim, reg_param=0.5):
    super(MetaQDA_FB, self).__init__()
    self.reg_param = reg_param
    self.feature_dim = x_dim
    self.m = torch.nn.Parameter(torch.zeros(1, self.feature_dim))
    self.kappa = torch.nn.Parameter(torch.tensor(0.1))
    self.nu = torch.nn.Parameter(torch.tensor(self.feature_dim, dtype=torch.float32))
    self.triu_diag = torch.nn.Parameter(torch.ones(self.feature_dim))
    self.triu_lower = torch.nn.Parameter(torch.eye(self.feature_dim))
    self.triu_lower_mask = torch.triu(torch.ones(self.feature_dim, self.feature_dim), diagonal=1).t().to(DEVICE)

  def fit_image_label(self, X, y):
    self.classes = np.unique(y.cpu())
    self.mu = [None for i in self.classes]
    self.sigma_inv = [None for i in self.classes]
    self.biases = [None for i in self.classes]
    self.common_part = [None for i in self.classes]
    self.lower_triu = torch.diag(torch.abs(self.triu_diag)) + self.triu_lower * self.triu_lower_mask

    kappa_ = torch.abs(self.kappa) + 1e-6
    nu_ = torch.clamp(self.nu, min=self.feature_dim - 1 + 1e-6)
    for j in self.classes:
      X_j = X[y == j]
      N_j = X_j.shape[0]
      self.mu[j] = kappa_.div(kappa_ + N_j) * self.m + torch.div(N_j, kappa_ + N_j) * torch.mean(X_j, dim=0,
                                                                                                 keepdim=True)
      sigma_j_no_scale = self_outer(self.lower_triu) + sum_outer(X_j) + kappa_ * self_outer(self.m) - (
                kappa_ + N_j) * self_outer(self.mu[j])
      sigma_j = torch.div(sigma_j_no_scale, (nu_ + N_j - self.feature_dim + 1) * (kappa_ + N_j)) * (kappa_ + N_j + 1)
      self.sigma_inv[j] = self.regularize(torch.inverse(sigma_j))
      self.common_part[j] = nu_ + N_j + 1 - self.feature_dim
      self.biases[j] = torch.lgamma(0.5 * (self.common_part[j] + self.feature_dim)) - torch.lgamma(
        0.5 * self.common_part[j]) - 0.5 * self.feature_dim * torch.log(self.common_part[j]) - 0.5 * torch.logdet(
        sigma_j)

  def predict(self, X):
    predicts_matrix = []
    for i in range(X.shape[0]):
      neg_distrances = []
      for mu, sigma_inv, bias, common_part in zip(self.mu, self.sigma_inv, self.biases, self.common_part):
        neg_distrances.append(bias - 0.5 * (common_part + self.feature_dim) * torch.log(
          1.0 + (1.0 / common_part) * self.compute_distance(X[i, :], mu, sigma_inv)))
      predicts_matrix.append(torch.cat(neg_distrances, dim=1))
    predicts_matrix = torch.cat(predicts_matrix, dim=0)
    return predicts_matrix

  def compute_distance(self, x, mu, sigma_inv):
    diff = x - mu
    gaussian_dist = torch.mm(torch.mm(diff, sigma_inv), diff.t())
    return gaussian_dist

  def regularize(self, sigma):
    return (1 - self.reg_param) * sigma + self.reg_param * torch.eye(self.feature_dim).to(DEVICE)


### common basic function
def self_outer(x):
  dim1, dim2 = x.shape
  if dim1 >= dim2:
    return torch.mm(x, x.t())
  elif dim1 < dim2:
    return torch.mm(x.t(), x)


def mean_outer(x):
  feature_dim = x.shape[1]
  S_ = torch.zeros((feature_dim, feature_dim), dtype=torch.float32).to(DEVICE)
  for index, a_v in enumerate(x):
    S_ += torch.mm(a_v.unsqueeze(1), a_v.unsqueeze(0))

  return S_.div(index + 1)


def sum_outer(x):
  feature_dim = x.shape[1]
  S_ = torch.zeros((feature_dim, feature_dim), dtype=torch.float32).to(DEVICE)
  for a_v in x:
    S_ += torch.mm(a_v.unsqueeze(1), a_v.unsqueeze(0))

  return S_


# A wrapper around MetaQDA implementations that exposes methods more in line with our codebase.
class MetaQDA(nn.Module):
  def __init__(self,
               feature_extractor,
               fe_dim,
               fe_dtype,
               device=torch.device('cuda:0'),
               variant='FB',
               reg_param=0.5):
    super(MetaQDA, self).__init__()
    self.variant = variant
    if self.variant == 'FB':
      self.meta_qda = MetaQDA_FB(fe_dim, reg_param)
    elif self.variant == 'MAP':
      self.meta_qda = MetaQDA_MAP(fe_dim, reg_param)

    self.feature_extractor = feature_extractor
    self.fe_dim = fe_dim
    self.fe_dtype = fe_dtype
    self.device = device

  def get_feature_vector(self, inp):
    batch_size = inp.size(0)
    origin_dtype = inp.dtype
    if origin_dtype != self.fe_dtype:
      inp = inp.to(self.fe_dtype)
      # print('Cast inputs to fe dtype', self.fe_dtype)
    feature_map = self.feature_extractor(inp)
    if feature_map.dtype != origin_dtype:
      feature_map = feature_map.to(origin_dtype)
    feature_vector = feature_map.view(batch_size, self.fe_dim)

    return feature_vector

  def forward(self, inp, labels, way, shot):
    # Assume that the first way*shot are the support, and queries are after that.
    with torch.no_grad():
      features = self.get_feature_vector(inp)

    # going to be shape (n_support, d)
    support_features = features[:way * shot, :]
    query_features = features[way * shot:, :]

    support_labels = labels

    exception = False
    try:
      self.meta_qda.fit_image_label(support_features, support_labels)
    except Exception as e:
      print(f'Exception: {e} has been raised.')
      print(f'support labels: {support_labels}')
      print(f'query_features shape: {query_features.shape}')
      print(f'support_features shape: {support_features.shape}')
      print(f'labels shape: {support_labels.shape}')
      exception = True
      raise Exception("Caught in first exception")
    if exception:
      raise Exception("caught in second exception")



    prediction = self.meta_qda.predict(query_features)
    # prediction has shape (num_query, way)
    return prediction

  def meta_test(self, inp, way, shot, query_shot):
    features = self.get_feature_vector(inp)
    support_features = features[:way * shot, :]
    query_features = features[way * shot:, :]
    support_labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device)

    self.meta_qda.fit_image_label(support_features, support_labels)
    prediction = self.meta_qda.predict(query_features)
    _, max_index = torch.max(prediction, 1)
    return max_index
