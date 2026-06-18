# LENS: Learning Explainable Neuro-Symbolic Representations of Narratives

**SemEval-2026 Task 4 — Narrative Story Similarity and Narrative Representation Learning**

A neuro-symbolic system that decomposes stories into structured narrative graphs, learns graph-level embeddings via a Heterogeneous Graph Neural Network, and fuses them with semantic text embeddings for narrative similarity.

## Architecture

```
                        Story Text
                       /          \
                      /            \
            LLM Extraction      Gemini Encoder
           (GPT-OSS:20b)       (gemini-embedding-001)
                |                      |
    Themes, Actions, Outcomes     Text Embedding t
                |                   (2048D, normalized)
      Graph Construction               |
    (Heterogeneous Narrative)           |
                |                       |
         all-MiniLM-L6-v2              |
          (384D node init)              |
                |                       |
           HeteroGNN                    |
     (SAGEConv, 3 layers)              |
                |                       |
      Graph Embedding g                 |
       (2048D, normalized)              |
                \                      /
                 \                    /
              Neuro-Symbolic Fusion
              z = (g_norm + t_norm) / 2
                        |
                  Cosine Similarity
```

## Pipeline Stages

| Stage | Script | Description | Requires |
|-------|--------|-------------|----------|
| 1. Preprocess | `preprocess.py` | Extract narrative components, build graphs, convert to PyG | Ollama |
| 2. Train | `train.py` | Base HeteroGNN training with triplet loss | GPU (optional) |
| 3. Gemini Embeddings | `generate_gemini_embeddings.py` | Generate text embeddings via Gemini API | API key |
| 4. Adapt | `adapt.py` | CORAL domain adaptation + recursive pseudo-labeling | - |
| 5. Inference | `infer.py` | Fuse graph + text embeddings, generate predictions | - |

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- CUDA-capable GPU (optional, CPU works but is slower)

### Installation

```bash
pip install -r requirements.txt

# Pull required Ollama models
ollama pull gpt-oss:20b        # Narrative extraction
ollama pull all-minilm:33m     # Node embeddings (384D)
```

### Environment Variables

Create a `.env` file in this directory:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Quick Start

```bash
# 1. Place training data in data/
cp synthetic_data_for_contrastive_learning.jsonl data/

# 2. Preprocess: extract narratives and build graphs
python preprocess.py

# 3. Train base HeteroGNN
python train.py

# 4. Generate Gemini text embeddings
python generate_gemini_embeddings.py --input data/test_track_a.jsonl --output outputs/gemini_embeddings.npy

# 5. Domain adaptation + pseudo-labeling
python adapt.py \
    --checkpoint checkpoints/hetero_gnn_trained.pt \
    --dev-data preprocessed/dev_pyg.pkl \
    --dev-labels data/dev_track_a.jsonl \
    --target-data preprocessed/test_pyg.pkl \
    --track-a-data preprocessed/test_pyg.pkl

# 6. Inference with fusion
python infer.py \
    --checkpoint checkpoints/hetero_gnn_adapted.pt \
    --data preprocessed/test_pyg.pkl \
    --gemini outputs/gemini_embeddings.npy \
    --output outputs/predictions.jsonl
```

## Project Structure

```
.
├── config.py                      # Central configuration
├── preprocess.py                  # Narrative extraction + graph construction
├── train.py                       # Base GNN training (triplet loss)
├── adapt.py                       # CORAL adaptation + pseudo-labeling
├── infer.py                       # Inference with neuro-symbolic fusion
├── generate_gemini_embeddings.py  # Gemini batch embedding generation
├── hyperparameter_tuning.py       # Grid search over GNN hyperparameters
│
├── models/
│   └── gnn.py                     # HeteroGNN architecture + PyG conversion
│
├── utils/
│   ├── ollama_client.py           # Ollama API wrapper
│   ├── extraction.py              # Narrative component extraction (Pydantic)
│   ├── graph_builder.py           # Heterogeneous graph construction
│   ├── data_loader.py             # PyG dataset + DataLoader
│   ├── losses.py                  # Soft-margin triplet + CORAL loss
│   └── augmentation.py            # Graph augmentation operations
│
├── data/                          # Training data (JSONL triplets)
├── preprocessed/                  # Cached extractions, graphs, PyG data
├── checkpoints/                   # Saved model weights
└── outputs/                       # Predictions and embeddings
```

## Technical Details

### Narrative Graph Structure

Each story is decomposed into a heterogeneous directed graph with four node types:

| Node Type | Description | Initialization |
|-----------|-------------|----------------|
| **Narrative** | Central story node | Summary embedding (384D) |
| **Theme** | Abstract conceptual ideas | Theme text embedding |
| **Action** | Sequential events | Action text embedding |
| **Outcome** | Final state / resolution | Outcome text embedding |

**Edge types** encode structural relationships:
- `theme_supports_story` — Theme grounds into the narrative
- `action_starts_story` / `action_ends_story` — Temporal anchoring
- `next_action` / `prev_action` — Bidirectional chronological flow
- `action_leads_to_outcome` — Causal relation
- `outcome_reflects_story` — Resolution grounding

### GNN Architecture

```
Input (384D all-MiniLM)
  → Per-node LayerNorm
  → Linear Projection (384 → 512)
  → 3x HeteroConv(SAGEConv, 512D) with residual connections
  → Narrative node readout (CLS-token analogue)
  → Linear Projection (512 → 2048D)
```

### Training Strategy

1. **Base training**: Soft-margin triplet loss on 1,900 synthetic story triplets
2. **CORAL adaptation**: Aligns covariance matrices between source (dev) and target (test) embedding distributions
3. **Pseudo-labeling**: Recursive rounds of confidence-based triplet mining under weak supervision, with consistency regularization via graph augmentation

### Fusion

Normalized averaging of graph and text embeddings preserves additive decomposability:

```
z = (normalize(graph_emb) + normalize(gemini_emb)) / 2
```

No post-fusion normalization is applied, maintaining independent contributions from both modalities.

## Contributors

- Rahul Krishnan ([@rahulk98](https://github.com/rahulk98))
- Sanjana Deshpande ([@ndsanjana](https://github.com/ndsanjana))
