# LENS: Learning Explainable Neuro-Symbolic Representations of Narratives

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.x-orange.svg)](https://pyg.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**SemEval-2026 Task 4: Narrative Story Similarity and Narrative Representation Learning**

Given two stories, LENS decides whether they share the same underlying narrative. It uses an LLM to decompose each story into its themes, actions, and outcomes, builds a heterogeneous narrative graph from those components, learns a graph-level embedding with a Graph Neural Network, and fuses that structural signal with a semantic text embedding. Because the fusion is plain additive averaging, the final similarity score decomposes cleanly into how much the structure channel and the text channel each contributed, so the result stays interpretable rather than being an opaque dense vector.

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

## Results

The fused representation outperforms either modality alone on a pairwise narrative-similarity task, and the fusion stays fully interpretable at the modality level.

### Performance

| Evaluation | Metric | Score |
|------------|--------|-------|
| Pairwise similarity (anchor vs. two candidates) | Accuracy | **72.5%** |
| Global cosine-similarity scoring | Accuracy | 64.75% |
| Cross-lingual generalization (Tell-Me-Again, 500 triples across de/en/es/fr/it) | Precision | **95%** |

Fusion beats either channel in isolation: the text channel rescues low-confidence graph decisions, and margin-sign agreement between the two channels is 89.8%.

### Interpretability

Omitting post-fusion normalization makes the fused cosine decompose exactly into additive terms, `S_fused = S_GG + S_TT + S_cross`:

- **Exact additive decomposition** holds at R^2 = 1.0.
- **Negligible cross term:** S_cross is ~0.0015 (+/- 0.005) against S_GG ~0.223 and S_TT ~0.133, so the two modalities contribute independently.
- **Near-orthogonal subspaces:** principal-angle analysis gives 85.4 degrees globally and 89.8 degrees per-sample, confirming the channels carry complementary information.
- **Rank asymmetry:** graph embeddings are effectively rank-1 (96.9% of variance in a single component), while text embeddings need 439 components to reach 95% variance.
- **Implicit confidence-gating with no learned parameters:** across 400 test samples the graph margin dominates 327 decisions, and the text channel rescues the remaining 73 where the graph signal is weak.
- **Which components drive the readout** (Frobenius-norm inspection of trained SAGEConv weights, reported as correlational): Action -> Story edges carry the strongest learned signal (71.2), followed by Outcome -> Story (69.8) and Theme -> Story (67.1).

### Limitations

- Graph quality depends entirely on upstream LLM extraction, which was not evaluated independently of end-task performance.
- The GNN was trained on only 1,900 synthetic triplets; transfer to substantially different narrative domains is untested.
- Weight-norm attributions are correlational; no edge-type ablations were run to confirm causality.
- The single narrative-node readout can bottleneck complex subplot structures.

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
