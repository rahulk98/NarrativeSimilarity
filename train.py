"""
Base training: HeteroGNN with soft-margin triplet loss on synthetic contrastive data.
No Ollama needed -- uses preprocessed PyG data.
"""

import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import Config
from models.gnn import HeteroGNN, get_metadata_from_data
from utils.losses import TripletLossWithSoftMargin
from utils.data_loader import create_dataloaders


class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.counter = 0
        self.best_value = None

    def __call__(self, value: float) -> bool:
        if self.best_value is None or value > self.best_value + 1e-4:
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Training", leave=False):
        batch_loss = 0.0
        for anchor, positive, negative in batch:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_emb = model(anchor.x_dict, anchor.edge_index_dict)
            p_emb = model(positive.x_dict, positive.edge_index_dict)
            n_emb = model(negative.x_dict, negative.edge_index_dict)

            loss = criterion(a_emb.unsqueeze(0), p_emb.unsqueeze(0), n_emb.unsqueeze(0))
            batch_loss += loss

        batch_loss = batch_loss / len(batch)
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in dataloader:
        for anchor, positive, negative in batch:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_emb = model(anchor.x_dict, anchor.edge_index_dict)
            p_emb = model(positive.x_dict, positive.edge_index_dict)
            n_emb = model(negative.x_dict, negative.edge_index_dict)

            loss = criterion(a_emb.unsqueeze(0), p_emb.unsqueeze(0), n_emb.unsqueeze(0))
            total_loss += loss.item()
            n_batches += 1

            # Ranking accuracy
            a_n = F.normalize(a_emb, p=2, dim=0)
            p_n = F.normalize(p_emb, p=2, dim=0)
            n_n = F.normalize(n_emb, p=2, dim=0)
            sim_pos = torch.dot(a_n, p_n).item()
            sim_neg = torch.dot(a_n, n_n).item()
            if sim_pos > sim_neg:
                correct += 1
            total += 1

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def train_gnn(config=Config):
    config.validate()
    config.print_config()
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # Load data
    train_loader, val_loader = create_dataloaders(config)

    # Get metadata from first sample
    first_batch = next(iter(train_loader))
    sample_anchor = first_batch[0][0]
    metadata = get_metadata_from_data(sample_anchor)
    print(f"Graph metadata: {len(metadata[0])} node types, {len(metadata[1])} edge types")

    # Initialize model
    model = HeteroGNN(
        in_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        out_channels=config.EMBEDDING_DIM,
        num_layers=config.NUM_GNN_LAYERS,
        metadata=metadata,
        aggr=config.GNN_AGGR,
        dropout=config.DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = TripletLossWithSoftMargin()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

    best_acc = 0.0
    best_state = None

    print(f"\nTraining for {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.GRADIENT_CLIP)

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | LR: {lr:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save best model
    if best_state:
        model.load_state_dict(best_state)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
        "config": {
            "input_channels": config.INPUT_CHANNELS,
            "hidden_channels": config.HIDDEN_CHANNELS,
            "embedding_dim": config.EMBEDDING_DIM,
            "num_gnn_layers": config.NUM_GNN_LAYERS,
            "dropout": config.DROPOUT,
            "gnn_aggr": config.GNN_AGGR,
        },
        "best_accuracy": best_acc,
    }

    save_path = config.CHECKPOINT_DIR / "hetero_gnn_trained.pt"
    torch.save(checkpoint, save_path)
    print(f"\nSaved best model (acc={best_acc:.1f}%) to {save_path}")

    # Save metadata
    meta_path = config.CHECKPOINT_DIR / "model_info.json"
    with open(meta_path, "w") as f:
        json.dump({"best_accuracy": best_acc, "config": checkpoint["config"]}, f, indent=2)

    return model


if __name__ == "__main__":
    train_gnn()
