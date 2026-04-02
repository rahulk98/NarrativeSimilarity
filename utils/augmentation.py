"""Graph augmentation operations for contrastive learning."""

import copy
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Base augmentation ops (used in on-the-fly training augmentation)
# ---------------------------------------------------------------------------

def drop_edges(data: HeteroData, drop_prob: float = 0.1) -> HeteroData:
    augmented = data.clone()
    for edge_type, edge_index in data.edge_index_dict.items():
        if edge_index.size(1) == 0:
            continue
        mask = torch.rand(edge_index.size(1)) > drop_prob
        if mask.sum() == 0:
            mask[0] = True
        augmented[edge_type].edge_index = edge_index[:, mask]
    return augmented


def add_node_noise(data: HeteroData, noise_std: float = 0.01) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        if features.size(0) > 0:
            augmented[node_type].x = features + torch.randn_like(features) * noise_std
    return augmented


def node_feature_masking(data: HeteroData, mask_prob: float = 0.1) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        if features.size(0) > 0:
            mask = torch.rand_like(features) > mask_prob
            augmented[node_type].x = features * mask
    return augmented


def feature_shuffling(data: HeteroData, shuffle_ratio: float = 0.2) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        if features.size(0) == 0 or features.size(1) == 0:
            continue
        num_to_shuffle = max(1, int(features.size(1) * shuffle_ratio))
        indices = torch.randperm(features.size(1))[:num_to_shuffle]
        for idx in indices:
            perm = torch.randperm(features.size(0))
            augmented[node_type].x[:, idx] = features[perm, idx]
    return augmented


def attribute_dropout(data: HeteroData, dropout_prob: float = 0.1) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        if features.size(0) > 0:
            mask = torch.rand_like(features) > dropout_prob
            scale = 1.0 / (1.0 - dropout_prob)
            augmented[node_type].x = features * mask * scale
    return augmented


def random_node_permutation(data: HeteroData, permute_ratio: float = 0.5) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        num_nodes = features.size(0)
        if num_nodes <= 1:
            continue
        n = max(2, int(num_nodes * permute_ratio))
        indices = torch.randperm(num_nodes, device=features.device)[:n]
        perm = indices[torch.randperm(len(indices), device=features.device)]
        augmented[node_type].x[indices] = features[perm]
    return augmented


def feature_scaling(data: HeteroData, scale_range: tuple = (0.8, 1.2)) -> HeteroData:
    augmented = data.clone()
    for node_type, features in data.x_dict.items():
        if features.size(0) == 0:
            continue
        lo, hi = scale_range
        scales = torch.rand(features.size(0), 1, device=features.device) * (hi - lo) + lo
        augmented[node_type].x = features * scales
    return augmented


AUGMENTATION_FNS = {
    "edge_dropout": lambda d, cfg: drop_edges(d, cfg.EDGE_DROP_PROB),
    "node_noise": lambda d, cfg: add_node_noise(d, cfg.NODE_NOISE_STD),
    "node_masking": lambda d, cfg: node_feature_masking(d, cfg.NODE_MASK_PROB),
    "feature_shuffling": lambda d, cfg: feature_shuffling(d, cfg.FEATURE_SHUFFLE_RATIO),
    "attribute_dropout": lambda d, cfg: attribute_dropout(d, cfg.ATTRIBUTE_DROPOUT_PROB),
    "random_node_permutation": lambda d, cfg: random_node_permutation(d, cfg.NODE_PERMUTE_RATIO),
    "feature_scaling": lambda d, cfg: feature_scaling(d, cfg.FEATURE_SCALE_RANGE),
}


def apply_random_augmentation(data: HeteroData, config) -> HeteroData:
    """Apply 1-3 random augmentation methods to a graph."""
    methods = config.AUGMENTATION_METHODS
    n = random.randint(1, min(3, len(methods)))
    selected = random.sample(methods, n)
    for method in selected:
        fn = AUGMENTATION_FNS.get(method)
        if fn:
            data = fn(data, config)
    return data


# ---------------------------------------------------------------------------
# Augmentation + consistency loss for pseudo-labeling
# ---------------------------------------------------------------------------

def augment_graph(data: HeteroData, node_drop_rate=0.1, edge_drop_rate=0.15, feature_noise_std=0.05):
    """Augment graph with node dropping, edge perturbation, and feature noise."""
    aug = copy.deepcopy(data)

    for node_type in aug.node_types:
        if node_type in aug.x_dict and aug.x_dict[node_type] is not None:
            x = aug.x_dict[node_type]
            aug.x_dict[node_type] = x + torch.randn_like(x) * feature_noise_std

    for edge_type in aug.edge_types:
        if edge_type in aug.edge_index_dict:
            ei = aug.edge_index_dict[edge_type]
            if ei is not None and ei.numel() > 0:
                keep = torch.rand(ei.size(1), device=ei.device) > edge_drop_rate
                aug.edge_index_dict[edge_type] = ei[:, keep]

    for node_type in aug.node_types:
        if node_type in aug.x_dict and aug.x_dict[node_type] is not None:
            x = aug.x_dict[node_type]
            drop = torch.rand(x.size(0), device=x.device) < node_drop_rate
            aug.x_dict[node_type][drop] = 0

    return aug


def consistency_loss(emb1, emb2):
    """Cosine-similarity based consistency loss: 1 - cos(emb1, emb2)."""
    return 1 - torch.dot(F.normalize(emb1, p=2, dim=0), F.normalize(emb2, p=2, dim=0))
