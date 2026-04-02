"""
Simplified hyperparameter grid search for HeteroGNN.
Searches over key architecture and training parameters.
"""

import gc
import json
import itertools
import random
from datetime import datetime

import numpy as np
import torch

from config import Config
from models.gnn import HeteroGNN, get_metadata_from_data
from utils.losses import TripletLossWithSoftMargin
from utils.data_loader import create_dataloaders
from train import train_one_epoch, validate, set_seed


PARAM_GRID = {
    "learning_rate": [0.0005, 0.001],
    "hidden_channels": [256, 512],
    "num_layers": [2, 3],
    "dropout": [0.1, 0.2],
    "batch_size": [32, 64],
}


def run_trial(params, config, train_loader, val_loader, metadata, device):
    """Run a single training trial with given hyperparameters."""
    set_seed(config.SEED)

    model = HeteroGNN(
        in_channels=config.INPUT_CHANNELS,
        hidden_channels=params["hidden_channels"],
        out_channels=config.EMBEDDING_DIM,
        num_layers=params["num_layers"],
        metadata=metadata,
        aggr=config.GNN_AGGR,
        dropout=params["dropout"],
    ).to(device)

    criterion = TripletLossWithSoftMargin()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params["learning_rate"], weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    best_acc = 0.0
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.GRADIENT_CLIP)

        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            best_acc = max(best_acc, val_acc)

        scheduler.step()

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_acc


def main():
    config = Config
    config.validate()
    device = torch.device(config.DEVICE)

    train_loader, val_loader = create_dataloaders(config)
    first_batch = next(iter(train_loader))
    metadata = get_metadata_from_data(first_batch[0][0])

    # Generate all parameter combinations
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    print(f"Running {len(combos)} trials...")

    results = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        print(f"\nTrial {i+1}/{len(combos)}: {params}")

        # Temporarily override config batch size
        original_bs = config.BATCH_SIZE
        config.BATCH_SIZE = params["batch_size"]
        tl, vl = create_dataloaders(config)
        config.BATCH_SIZE = original_bs

        try:
            acc = run_trial(params, config, tl, vl, metadata, device)
            params["best_accuracy"] = acc
            results.append(params)
            print(f"  -> Accuracy: {acc:.1f}%")
        except Exception as e:
            print(f"  -> Failed: {e}")
            params["error"] = str(e)
            results.append(params)

    # Sort by accuracy
    results.sort(key=lambda x: x.get("best_accuracy", 0), reverse=True)

    # Save results
    output_path = config.OUTPUT_DIR / f"hp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\nTop 3 configurations:")
    for r in results[:3]:
        print(f"  Acc: {r.get('best_accuracy', 0):.1f}% | {r}")


if __name__ == "__main__":
    main()
