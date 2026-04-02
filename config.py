"""Central configuration for the LENS pipeline."""

import os
import torch
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"
    PREPROCESSED_DIR = BASE_DIR / "preprocessed"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    OUTPUT_DIR = BASE_DIR / "outputs"

    # Ollama models
    EXTRACTION_MODEL = "gpt-oss:20b"
    EMBEDDING_MODEL = "all-minilm:33m"
    EMBEDDING_DIM_INPUT = 384  # all-MiniLM-L6-v2 output dimension

    # GNN architecture
    INPUT_CHANNELS = 384
    HIDDEN_CHANNELS = 512
    EMBEDDING_DIM = 2048
    NUM_GNN_LAYERS = 3
    DROPOUT = 0.1
    GNN_AGGR = "sum"

    # Base training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20
    TRIPLET_MARGIN = 0.3
    GRADIENT_CLIP = 0.5
    EARLY_STOPPING_PATIENCE = 5
    SEED = 922

    # LR scheduler (StepLR)
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.5

    # Augmentation (on-the-fly during base training)
    USE_AUGMENTATION = True
    AUGMENTATION_METHODS = [
        "edge_dropout", "node_noise", "node_masking",
        "feature_shuffling", "attribute_dropout",
        "random_node_permutation", "feature_scaling",
    ]
    EDGE_DROP_PROB = 0.25
    NODE_NOISE_STD = 0.1
    NODE_MASK_PROB = 0.2
    FEATURE_SHUFFLE_RATIO = 0.3
    ATTRIBUTE_DROPOUT_PROB = 0.25
    NODE_PERMUTE_RATIO = 0.5
    FEATURE_SCALE_RANGE = (0.7, 1.3)

    # Domain adaptation (CORAL)
    CORAL_WEIGHT = 0.3
    ADAPTATION_EPOCHS = 20
    ADAPTATION_LR = 5e-6

    # Pseudo-labeling (recursive)
    PSEUDO_ROUNDS = 15
    PSEUDO_EPOCHS = 10
    INITIAL_PSEUDO_EPOCHS = 10
    PSEUDO_MARGIN = 0.15
    PSEUDO_MARGIN_STEP = 0.05
    PSEUDO_TRAIN_MARGIN = 0.2
    PSEUDO_MIN_NEW = 2
    PSEUDO_MAX_TRIPLETS = 15000
    PSEUDO_BATCH_SIZE = 16
    PSEUDO_LR = 5e-5
    CONSISTENCY_WEIGHT = 0.1

    # Augmentation for pseudo-labeling
    PSEUDO_NODE_DROP_RATE = 0.1
    PSEUDO_EDGE_DROP_RATE = 0.15
    PSEUDO_FEATURE_NOISE_STD = 0.05

    # Gemini embeddings
    GEMINI_MODEL = "gemini-embedding-001"
    GEMINI_DIM = 2048
    GEMINI_BATCH_SIZE = 50

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    CONTRASTIVE_DATA = "synthetic_data_for_contrastive_learning.jsonl"
    VAL_SPLIT = 0.1

    @classmethod
    def validate(cls):
        for d in [cls.DATA_DIR, cls.PREPROCESSED_DIR, cls.CHECKPOINT_DIR, cls.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("=== LENS Configuration ===")
        print(f"  Device: {cls.DEVICE}")
        print(f"  GNN: {cls.INPUT_CHANNELS}D -> {cls.HIDDEN_CHANNELS}D x{cls.NUM_GNN_LAYERS} -> {cls.EMBEDDING_DIM}D")
        print(f"  Training: batch={cls.BATCH_SIZE}, lr={cls.LEARNING_RATE}, epochs={cls.NUM_EPOCHS}")
        print(f"  Adaptation: CORAL weight={cls.CORAL_WEIGHT}, epochs={cls.ADAPTATION_EPOCHS}")
        print(f"  Pseudo-labeling: rounds={cls.PSEUDO_ROUNDS}, margin={cls.PSEUDO_MARGIN}")
        print(f"  Fusion: GNN({cls.EMBEDDING_DIM}D) + Gemini({cls.GEMINI_DIM}D)")
        print("=" * 30)
