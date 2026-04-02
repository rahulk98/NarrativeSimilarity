"""
Inference: load model, generate embeddings, fuse with Gemini, predict.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from models.gnn import HeteroGNN


def load_model(checkpoint_path: str, device: str):
    """Load trained HeteroGNN from checkpoint."""
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
    model.eval()
    return model


@torch.no_grad()
def generate_graph_embedding(model, pyg_data, device: str) -> np.ndarray:
    """Forward pass through GNN, returns L2-normalized embedding."""
    pyg_data = pyg_data.to(device)
    emb = model(pyg_data.x_dict, pyg_data.edge_index_dict)
    emb = F.normalize(emb, p=2, dim=0)
    return emb.cpu().numpy()


def fuse_embeddings(graph_emb: np.ndarray, gemini_emb: np.ndarray) -> np.ndarray:
    """
    Neuro-symbolic fusion: z = (normalize(g) + normalize(t)) / 2
    No post-fusion normalization (preserves additive decomposability).
    """
    g_norm = graph_emb / (np.linalg.norm(graph_emb) + 1e-8)
    t_norm = gemini_emb / (np.linalg.norm(gemini_emb) + 1e-8)
    return (g_norm + t_norm) / 2


def predict_triplets(
    model,
    triplet_data: list,
    gemini_embeddings: np.ndarray,
    device: str,
) -> list:
    """
    Generate predictions for Track A triplets.

    Args:
        model: Trained HeteroGNN
        triplet_data: List of PyG triplet dicts
        gemini_embeddings: Array of shape (N*3, dim) ordered as
                          [anchor_0, text_a_0, text_b_0, anchor_1, ...]
        device: torch device string
    """
    predictions = []

    for i, triplet in enumerate(tqdm(triplet_data, desc="Predicting")):
        anchor_g = triplet.get("anchor_data") or triplet.get("anchor")
        text_a_g = triplet.get("text_a_data") or triplet.get("similar_data") or triplet.get("similar")
        text_b_g = triplet.get("text_b_data") or triplet.get("dissimilar_data") or triplet.get("dissimilar")

        # Graph embeddings
        a_graph = generate_graph_embedding(model, anchor_g, device)
        ta_graph = generate_graph_embedding(model, text_a_g, device)
        tb_graph = generate_graph_embedding(model, text_b_g, device)

        # Fuse with Gemini
        if gemini_embeddings is not None and i * 3 + 2 < len(gemini_embeddings):
            a_emb = fuse_embeddings(a_graph, gemini_embeddings[i * 3])
            ta_emb = fuse_embeddings(ta_graph, gemini_embeddings[i * 3 + 1])
            tb_emb = fuse_embeddings(tb_graph, gemini_embeddings[i * 3 + 2])
        else:
            a_emb, ta_emb, tb_emb = a_graph, ta_graph, tb_graph

        # Cosine similarity
        sim_a = float(np.dot(a_emb, ta_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(ta_emb) + 1e-8))
        sim_b = float(np.dot(a_emb, tb_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(tb_emb) + 1e-8))

        predictions.append({
            "triplet_id": triplet.get("triplet_id", i),
            "text_a_is_closer": sim_a > sim_b,
            "sim_anchor_text_a": round(sim_a, 6),
            "sim_anchor_text_b": round(sim_b, 6),
        })

    return predictions


def save_predictions(predictions: list, output_path: str):
    """Save predictions as JSONL."""
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Saved {len(predictions)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LENS inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hetero_gnn_trained.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to PyG triplet data (.pkl)")
    parser.add_argument("--gemini", type=str, default=None, help="Path to Gemini embeddings (.npy)")
    parser.add_argument("--output", type=str, default="outputs/predictions.jsonl")
    args = parser.parse_args()

    config = Config
    config.validate()
    device = config.DEVICE

    print("Loading model...")
    model = load_model(args.checkpoint, device)

    print("Loading data...")
    with open(args.data, "rb") as f:
        triplet_data = pickle.load(f)
    print(f"Loaded {len(triplet_data)} triplets")

    gemini_embs = None
    if args.gemini:
        gemini_embs = np.load(args.gemini)
        print(f"Loaded Gemini embeddings: {gemini_embs.shape}")

    predictions = predict_triplets(model, triplet_data, gemini_embs, device)

    # Accuracy if ground truth available
    correct = sum(1 for p in predictions if "text_a_is_closer" in p)
    print(f"Generated {len(predictions)} predictions")

    save_predictions(predictions, args.output)


if __name__ == "__main__":
    main()
