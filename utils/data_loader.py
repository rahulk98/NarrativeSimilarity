"""Dataset and DataLoader for triplet-based GNN training."""

import pickle
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData

from .augmentation import apply_random_augmentation


class TripletDataset(Dataset):
    """Loads preprocessed PyG triplets with optional on-the-fly augmentation."""

    def __init__(self, data_path: str, config=None, augment: bool = True):
        with open(data_path, "rb") as f:
            self.triplets = pickle.load(f)
        self.config = config
        self.augment = augment and config is not None and config.USE_AUGMENTATION

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        if isinstance(triplet, dict):
            anchor = triplet.get("anchor_data") or triplet.get("anchor")
            positive = triplet.get("similar_data") or triplet.get("similar")
            negative = triplet.get("dissimilar_data") or triplet.get("dissimilar")
        else:
            anchor, positive, negative = triplet

        if self.augment:
            anchor = apply_random_augmentation(anchor, self.config)
            positive = apply_random_augmentation(positive, self.config)
            negative = apply_random_augmentation(negative, self.config)

        return anchor, positive, negative


def collate_triplets(batch):
    """Custom collate: returns list of triplet tuples (no batching of HeteroData)."""
    return batch


def create_dataloaders(
    config,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and optional validation data loaders."""
    if train_path is None:
        # Auto-detect from preprocessed directory
        pkl_files = sorted(config.PREPROCESSED_DIR.glob("pyg_data*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No PyG data files found in {config.PREPROCESSED_DIR}")
        train_path = str(pkl_files[0])

    full_dataset = TripletDataset(train_path, config=config, augment=True)

    # Split into train/val
    n = len(full_dataset)
    n_val = max(1, int(n * config.VAL_SPLIT)) if n >= 2 else 0
    n_train = n - n_val

    indices = list(range(n))
    random.seed(config.SEED)
    random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(
        TripletDataset(train_path, config=config, augment=False),
        val_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_triplets,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_triplets,
        num_workers=0,
    ) if n_val > 0 else None

    print(f"Data: {n_train} train, {n_val} val triplets")
    return train_loader, val_loader
