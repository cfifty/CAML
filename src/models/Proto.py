import sys
import torch
import torch.nn as nn
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


class Proto(nn.Module):

    def __init__(self,
                 feature_extractor,
                 fe_dim,
                 fe_dtype,
                 device=torch.device('cuda:0'),
                 way=None,
                 shots=None,
                 **kwargs):

        super().__init__()
        self.feature_extractor = feature_extractor
        self.fe_dim = fe_dim
        self.fe_dtype = fe_dtype
        self.device = device
        self.cosine_sim = torch.nn.CosineSimilarity(dim=2)
        self.way = way
        self.shots = shots

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0),
                                 requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10),
                                      requires_grad=True)

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

    def forward(self, inp, support_labels, way, shot, flip_centroids=False):
        # Features will be |shot| contiguous elements.
        features = self.get_feature_vector(inp)
        b, d = features.shape

        # Compute the centroids over the "shot" dimension -- aggregates all points belonging to the same class together.
        support = features[:way * shot].reshape(-1, way, shot, d)
        # Centroids should have shape: (b, way, d)
        centroids = 1 / shot * torch.sum(support, dim=2)

        if flip_centroids:
            centroids = torch.flip(centroids, dims=(1, ))

        query = features[way * shot:].reshape(-1, 1, d)

        # Logits should have shape: [b, way]
        # PMF uses raw cosine similarity + adds a scale + bias term to compose logits => for use in cross-entropy loss.
        logits = self.scale_cls * (
            self.cosine_sim(query, centroids).squeeze() + self.bias)
        return logits

    def meta_test(self, inp, way, shot, query_shot):
        feature_vector = self.get_feature_vector(inp)
        b, d = feature_vector.shape
        support_features = feature_vector[:way * shot]
        query_features = feature_vector[way * shot:]

        support_features = support_features.reshape(1, way, shot, d)
        centroids = 1 / shot * torch.sum(support_features, dim=2)

        query_features = query_features.reshape(-1, 1, d)
        logits = self.scale_cls * (
            self.cosine_sim(query_features, centroids).squeeze() + self.bias)
        _, max_index = torch.max(logits, 1)
        return max_index
