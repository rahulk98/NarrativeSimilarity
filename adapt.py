"""
Domain adaptation (CORAL) and recursive pseudo-labeling.
Runs after base training to adapt the model to target distribution.

Stages:
    1. CORAL alignment: Dev -> Test distribution via covariance alignment
    2. Track A pseudo-labeling: Recursive confidence-based triplet mining
    3. Track B pseudo-labeling: Contrastive pre-training on story pairs
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from models.gnn import HeteroGNN, get_metadata_from_data
from utils.losses import CORALLoss
from utils.augmentation import augment_graph, consistency_loss


def load_model(checkpoint_path: str, device: str):
    """Load a pre-trained HeteroGNN from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    metadata = ckpt["metadata"]

    model = HeteroGNN(
        in_channels=cfg["input_channels"],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["embedding_dim"],
        num_layers=cfg["num_gnn_layers"],
        metadata=metadata,
        aggr=cfg.get("gnn_aggr", "sum"),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, metadata


def get_pyg_data(entry):
    """Unwrap PyG data from a dict if needed."""
    if isinstance(entry, dict) and "pyg_data" in entry:
        return entry["pyg_data"]
    return entry


# ---------------------------------------------------------------------------
# Stage 1: CORAL Domain Adaptation
# ---------------------------------------------------------------------------

def run_domain_adaptation(model, dev_triplets, target_graphs, dev_labels, config):
    """Align dev distribution to target distribution using CORAL + triplet loss."""
    device = torch.device(config.DEVICE)
    coral_fn = CORALLoss()
    triplet_fn = torch.nn.TripletMarginLoss(margin=config.PSEUDO_TRAIN_MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ADAPTATION_LR)

    # Build labeled dev triplets
    dev_data = []
    for label in dev_labels:
        idx = label.get("triplet_id", len(dev_data))
        if idx < len(dev_triplets):
            t = dev_triplets[idx]
            a = get_pyg_data(t.get("anchor_data") or t.get("anchor"))
            b = get_pyg_data(t.get("text_a_data") or t.get("similar_data") or t.get("similar"))
            c = get_pyg_data(t.get("text_b_data") or t.get("dissimilar_data") or t.get("dissimilar"))
            if label.get("text_a_is_closer", True):
                dev_data.append((a, b, c))
            else:
                dev_data.append((a, c, b))

    # Pre-compute target embeddings
    target_embs = []
    model.eval()
    with torch.no_grad():
        for g in target_graphs[:min(900, len(target_graphs))]:
            g = g.to(device)
            emb = F.normalize(model(g.x_dict, g.edge_index_dict), p=2, dim=0)
            target_embs.append(emb)
    target_embs = torch.stack(target_embs)

    model.train()
    print(f"\nCORAL adaptation: {config.ADAPTATION_EPOCHS} epochs, weight={config.CORAL_WEIGHT}")

    for epoch in range(config.ADAPTATION_EPOCHS):
        total_triplet = 0.0
        total_coral = 0.0
        random.shuffle(dev_data)

        for i in range(0, len(dev_data), config.PSEUDO_BATCH_SIZE):
            batch = dev_data[i : i + config.PSEUDO_BATCH_SIZE]
            batch_triplet = 0.0
            dev_embs = []

            for a_g, b_g, c_g in batch:
                a_g, b_g, c_g = a_g.to(device), b_g.to(device), c_g.to(device)
                a_emb = model(a_g.x_dict, a_g.edge_index_dict)
                b_emb = model(b_g.x_dict, b_g.edge_index_dict)
                c_emb = model(c_g.x_dict, c_g.edge_index_dict)

                batch_triplet += triplet_fn(a_emb.unsqueeze(0), b_emb.unsqueeze(0), c_emb.unsqueeze(0))
                dev_embs.extend([F.normalize(a_emb, p=2, dim=0),
                                 F.normalize(b_emb, p=2, dim=0),
                                 F.normalize(c_emb, p=2, dim=0)])

            dev_stack = torch.stack(dev_embs)
            alignment = coral_fn(dev_stack, target_embs)
            loss = (batch_triplet / len(batch)) + config.CORAL_WEIGHT * alignment

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_triplet += batch_triplet.item() / len(batch)
            total_coral += alignment.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config.ADAPTATION_EPOCHS} | Triplet: {total_triplet:.4f} | CORAL: {total_coral:.4f}")

    print("CORAL adaptation complete.")


# ---------------------------------------------------------------------------
# Stage 2: Pseudo-Label Mining and Training
# ---------------------------------------------------------------------------

def mine_triplets(model, triplet_data, margin, max_triplets, seen_keys, device):
    """Mine pseudo-labeled triplets using similarity margin threshold."""
    model.eval()
    mined = []

    for idx, item in enumerate(triplet_data):
        try:
            a_g = get_pyg_data(item.get("anchor_data") or item.get("anchor")).to(device)
            b_g = get_pyg_data(item.get("text_a_data") or item.get("similar_data") or item.get("similar")).to(device)
            c_g = get_pyg_data(item.get("text_b_data") or item.get("dissimilar_data") or item.get("dissimilar")).to(device)

            with torch.no_grad():
                a = F.normalize(model(a_g.x_dict, a_g.edge_index_dict), p=2, dim=0)
                b = F.normalize(model(b_g.x_dict, b_g.edge_index_dict), p=2, dim=0)
                c = F.normalize(model(c_g.x_dict, c_g.edge_index_dict), p=2, dim=0)

            sim_b = torch.dot(a, b).item()
            sim_c = torch.dot(a, c).item()

            if abs(sim_b - sim_c) < margin:
                continue

            if sim_b > sim_c:
                key = (idx, "a")
                if key not in seen_keys:
                    mined.append((a_g.cpu(), b_g.cpu(), c_g.cpu()))
                    seen_keys.add(key)
            else:
                key = (idx, "b")
                if key not in seen_keys:
                    mined.append((a_g.cpu(), c_g.cpu(), b_g.cpu()))
                    seen_keys.add(key)

            if len(mined) >= max_triplets:
                break
        except Exception:
            continue

    model.train()
    return mined


def pseudo_label_train(model, all_triplets, config, num_epochs):
    """Train model on pseudo-labeled triplets with consistency regularization."""
    device = torch.device(config.DEVICE)
    loss_fn = torch.nn.TripletMarginLoss(margin=config.PSEUDO_TRAIN_MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.PSEUDO_LR)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        random.shuffle(all_triplets)

        for i in range(0, len(all_triplets), config.PSEUDO_BATCH_SIZE):
            batch = all_triplets[i : i + config.PSEUDO_BATCH_SIZE]
            batch_loss = 0.0

            for a_g, b_g, c_g in batch:
                a_g, b_g, c_g = a_g.to(device), b_g.to(device), c_g.to(device)
                a_emb = model(a_g.x_dict, a_g.edge_index_dict)
                b_emb = model(b_g.x_dict, b_g.edge_index_dict)
                c_emb = model(c_g.x_dict, c_g.edge_index_dict)
                triplet_loss = loss_fn(a_emb.unsqueeze(0), b_emb.unsqueeze(0), c_emb.unsqueeze(0))

                # Consistency regularization
                a_aug = augment_graph(a_g, config.PSEUDO_NODE_DROP_RATE, config.PSEUDO_EDGE_DROP_RATE, config.PSEUDO_FEATURE_NOISE_STD)
                b_aug = augment_graph(b_g, config.PSEUDO_NODE_DROP_RATE, config.PSEUDO_EDGE_DROP_RATE, config.PSEUDO_FEATURE_NOISE_STD)
                c_aug = augment_graph(c_g, config.PSEUDO_NODE_DROP_RATE, config.PSEUDO_EDGE_DROP_RATE, config.PSEUDO_FEATURE_NOISE_STD)

                a_aug_emb = model(a_aug.x_dict, a_aug.edge_index_dict)
                b_aug_emb = model(b_aug.x_dict, b_aug.edge_index_dict)
                c_aug_emb = model(c_aug.x_dict, c_aug.edge_index_dict)

                cons = (consistency_loss(a_emb, a_aug_emb) +
                        consistency_loss(b_emb, b_aug_emb) +
                        consistency_loss(c_emb, c_aug_emb)) / 3

                batch_loss += triplet_loss + config.CONSISTENCY_WEIGHT * cons

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f}")


def run_pseudo_labeling(model, triplet_data, config):
    """Recursive pseudo-label mining and training."""
    device = torch.device(config.DEVICE)
    seen_keys = set()
    all_triplets = []
    margin = config.PSEUDO_MARGIN

    # Initial mining
    print(f"\nInitial mining (margin={margin})...")
    initial = mine_triplets(model, triplet_data, margin, config.PSEUDO_MAX_TRIPLETS, seen_keys, device)
    print(f"Found {len(initial)} initial pseudo-labeled triplets")

    if len(initial) < config.PSEUDO_MIN_NEW:
        print("Not enough initial triplets. Skipping pseudo-labeling.")
        return

    all_triplets = initial
    print(f"Training on initial labels...")
    pseudo_label_train(model, all_triplets, config, config.INITIAL_PSEUDO_EPOCHS)

    # Recursive rounds
    print(f"\nRecursive pseudo-labeling ({config.PSEUDO_ROUNDS} rounds)...")
    for round_num in range(1, config.PSEUDO_ROUNDS + 1):
        new = mine_triplets(model, triplet_data, margin, config.PSEUDO_MAX_TRIPLETS, seen_keys, device)
        print(f"Round {round_num}: {len(new)} new triplets (total: {len(all_triplets) + len(new)})")

        if len(new) < config.PSEUDO_MIN_NEW:
            print("Stopping: not enough new pseudo-labels.")
            break

        all_triplets.extend(new)
        pseudo_label_train(model, all_triplets, config, config.PSEUDO_EPOCHS)

    print(f"Pseudo-labeling complete. Total triplets used: {len(all_triplets)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Domain adaptation and pseudo-labeling")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hetero_gnn_trained.pt")
    parser.add_argument("--target-data", type=str, help="Path to target (test) PyG data for CORAL")
    parser.add_argument("--dev-data", type=str, help="Path to dev PyG data")
    parser.add_argument("--dev-labels", type=str, help="Path to dev labels JSONL")
    parser.add_argument("--track-a-data", type=str, help="Path to Track A PyG triplets for pseudo-labeling")
    parser.add_argument("--skip-coral", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true")
    args = parser.parse_args()

    config = Config
    config.validate()
    device = torch.device(config.DEVICE)

    print("Loading pre-trained model...")
    model, metadata = load_model(args.checkpoint, config.DEVICE)

    # Stage 1: CORAL
    if not args.skip_coral and args.dev_data and args.target_data and args.dev_labels:
        with open(args.dev_data, "rb") as f:
            dev_triplets = pickle.load(f)
        with open(args.target_data, "rb") as f:
            target_raw = pickle.load(f)

        target_graphs = []
        if isinstance(target_raw, list):
            for item in target_raw:
                if isinstance(item, dict):
                    for key in ["anchor_data", "text_a_data", "text_b_data"]:
                        if key in item:
                            target_graphs.append(get_pyg_data(item[key]))
                else:
                    target_graphs.append(item)

        dev_labels = []
        with open(args.dev_labels, "r") as f:
            for line in f:
                if line.strip():
                    dev_labels.append(json.loads(line))

        run_domain_adaptation(model, dev_triplets, target_graphs, dev_labels, config)

    # Stage 2: Pseudo-labeling
    if not args.skip_pseudo and args.track_a_data:
        with open(args.track_a_data, "rb") as f:
            track_a = pickle.load(f)
        run_pseudo_labeling(model, track_a, config)

    # Save adapted model
    save_path = config.CHECKPOINT_DIR / "hetero_gnn_adapted.pt"
    save_payload = {
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
    }
    torch.save(save_payload, save_path)
    print(f"\nSaved adapted model to {save_path}")


if __name__ == "__main__":
    main()
