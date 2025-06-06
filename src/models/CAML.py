import sys
from typing import Literal
import torch

import torch.nn as nn

sys.path.append("/home/ubuntu/flowtolearn-D2NWG/CAML")

from src.models.TransformerEncoder import get_encoder


class CAML(nn.Module):

    def __init__(
        self,
        feature_extractor,
        fe_dim,
        fe_dtype,
        train_fe,
        encoder_size,
        num_classes: int,
        device=torch.device("cuda:0"),
        label_elmes=True,
        **kwargs
    ):

        super().__init__()
        self.feature_extractor = feature_extractor.to(device)
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
        self.transformer_encoder = get_encoder(
            encoder_size,
            image_dim=self.fe_dim,
            num_classes=num_classes,
            device=device,
            label_elmes=label_elmes,
            **kwargs
        )

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
        with torch.no_grad():
            features = self.get_feature_vector(inp)
        _, d = features.shape
        query = features[way * shot :].reshape(-1, 1, d)
        b, _, _ = query.shape

        # Repeat the support |B| times for each query example.
        support = features[: way * shot].reshape(1, way * shot, d).repeat(b, 1, 1)

        feature_sequences = torch.cat([query, support], dim=1)
        logits = self.transformer_encoder.forward_imagenet_v2(
            feature_sequences, labels, way, shot
        )
        return logits

    def meta_test(self, inp, way, shot, query_shot):
        """For evaluating typical Meta-Learning Datasets."""
        feature_vector = self.get_feature_vector(inp)
        support_features = feature_vector[: way * shot]
        query_features = feature_vector[way * shot :]
        b, d = query_features.shape

        # Reshape query and support to a sequence.
        support = support_features.reshape(1, way * shot, d).repeat(b, 1, 1)
        query = query_features.reshape(-1, 1, d)
        feature_sequences = torch.cat([query, support], dim=1)

        labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device)
        logits = self.transformer_encoder.forward_imagenet_v2(
            feature_sequences, labels, way, shot
        )
        _, max_index = torch.max(logits, 1)
        return max_index

    def encode(
        self,
        inp: torch.Tensor,
        way: int,
        shot: int,
        query_shot: int,
        encode_type: Literal["before_sequence_model", "after_sequence_model"],
        support_labels: torch.Tensor | None = None,
        features_as_input: bool = False,
        # combine_method: Literal["mean", "linear_proj", "pma", "none"] = "none",
    ) -> torch.Tensor:
        """

        Args:
            inp (torch.Tensor): input tensor of shape (batch_size, channels, height, width) (batch_size has to be way * (shot + query_shot))
            way (int): number of classes per task
            shot (int): number of support examples per class
            query_shot (int): number of query examples per class
            encode_type (str): type of encoding to be used
            support_labels (torch.Tensor): labels of the support images (num_support, )
            features_as_input (bool): If True, inputs is assumed to be a tensor of features, shape (batch_size, emb_dim)


        Returns:
            torch.Tensor: encoded CAML vector of the dataset
        """
        if features_as_input:
            feature_vector = inp
        else:
            feature_vector = self.get_feature_vector(inp)
        support_features = feature_vector[: way * shot]
        query_features = feature_vector[way * shot :]
        b, d = query_features.shape

        # Reshape query and support to a sequence.
        support = support_features.reshape(1, way * shot, d).repeat(b, 1, 1)
        query = query_features.reshape(-1, 1, d)
        feature_sequences = torch.cat([query, support], dim=1)

        if support_labels is not None:
            labels = support_labels
        else:
            # Create a label tensor for the support set if not provided.
            labels = torch.LongTensor([i // shot for i in range(shot * way)]).to(inp.device)

        if encode_type == "after_sequence_model":
            all_features = self.transformer_encoder.forward(feature_sequences, labels)
            query_features = all_features[:, 0, :]

            # # Average all the query features to get the dataset embedding.
            # if combine_method == "mean":
            #     dataset_embedding = query_features.mean(dim=0) # => (emb_size)
            # elif combine_method == "none":
            #     dataset_embedding = query_features # => (way * query_shot, emb_size)
            # elif combine_method == "linear_proj":

        elif encode_type == "before_sequence_model":
            raise NotImplementedError("Not implemented yet.")

        return query_features # => (way * query_shot, emb_size)
