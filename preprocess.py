"""
Preprocessing pipeline: story text -> narrative extraction -> graph -> PyG data.
Requires Ollama running with gpt-oss:20b and all-minilm:33m models.
"""

import json
import pickle
from pathlib import Path
from tqdm import tqdm

from config import Config
from utils.extraction import (
    extract_narrative_components,
    generate_story_summary,
    normalize_embeddings,
)
from utils.graph_builder import batch_embed_texts, create_narrative_graph
from models.gnn import nx_to_pyg_hetero


def load_triplets(data_path: str) -> list:
    """Load triplets from JSONL file."""
    triplets = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def extract_all_texts(extraction, summary: str) -> list:
    """Collect all unique texts from an extraction for batch embedding."""
    texts = [summary]
    texts.extend(extraction.abstract_theme)
    texts.extend(extraction.course_of_action)
    texts.extend(extraction.outcome)
    return texts


def process_story(story_text: str, config: Config) -> dict:
    """Extract components and generate summary for one story."""
    extraction = extract_narrative_components(story_text, config.EXTRACTION_MODEL)
    summary = generate_story_summary(story_text, config.EXTRACTION_MODEL)
    return {"text": story_text, "extraction": extraction, "summary": summary}


def main():
    config = Config
    config.validate()
    config.print_config()

    data_path = config.DATA_DIR / config.CONTRASTIVE_DATA
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print(f"Place your training data JSONL in {config.DATA_DIR}/")
        return

    print(f"\nLoading triplets from {data_path}...")
    raw_triplets = load_triplets(str(data_path))
    print(f"Loaded {len(raw_triplets)} triplets")

    # Step 1: Extract narrative components
    extractions_path = config.PREPROCESSED_DIR / "extractions.jsonl"
    checkpoint_path = config.PREPROCESSED_DIR / "extraction_checkpoint.jsonl"

    # Resume from checkpoint if available
    processed = []
    start_idx = 0
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            for line in f:
                if line.strip():
                    processed.append(json.loads(line))
        start_idx = len(processed)
        print(f"Resuming from checkpoint: {start_idx}/{len(raw_triplets)} triplets")

    print(f"\nStep 1: Extracting narrative components...")
    for i in tqdm(range(start_idx, len(raw_triplets)), initial=start_idx, total=len(raw_triplets)):
        triplet = raw_triplets[i]
        anchor_text = triplet.get("anchor_story") or triplet.get("anchor_text")
        similar_text = triplet.get("similar_story") or triplet.get("text_a")
        dissimilar_text = triplet.get("dissimilar_story") or triplet.get("text_b")

        try:
            anchor = process_story(anchor_text, config)
            similar = process_story(similar_text, config)
            dissimilar = process_story(dissimilar_text, config)

            entry = {
                "triplet_id": i,
                "anchor": anchor,
                "similar": similar,
                "dissimilar": dissimilar,
            }
            # Serialize extractions for JSON
            entry["anchor"]["extraction"] = entry["anchor"]["extraction"].model_dump()
            entry["similar"]["extraction"] = entry["similar"]["extraction"].model_dump()
            entry["dissimilar"]["extraction"] = entry["dissimilar"]["extraction"].model_dump()

            processed.append(entry)

            # Checkpoint every 100 triplets
            if (i + 1) % 100 == 0:
                with open(checkpoint_path, "w") as f:
                    for p in processed:
                        f.write(json.dumps(p) + "\n")
                print(f"  Checkpoint saved at {i + 1} triplets")
        except Exception as e:
            print(f"  Error processing triplet {i}: {e}")
            continue

    # Save final extractions
    with open(extractions_path, "w") as f:
        for p in processed:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(processed)} extractions to {extractions_path}")

    # Step 2: Batch embed all texts
    print(f"\nStep 2: Batch embedding all texts...")
    all_texts = set()
    for entry in processed:
        for role in ["anchor", "similar", "dissimilar"]:
            ext = entry[role]["extraction"]
            from utils.extraction import NarrativeExtractionOutput
            extraction = NarrativeExtractionOutput(**ext)
            all_texts.update(extract_all_texts(extraction, entry[role]["summary"]))

    all_texts = list(all_texts)
    print(f"  {len(all_texts)} unique texts to embed")
    embeddings_cache = batch_embed_texts(all_texts, config.EMBEDDING_MODEL, config.EMBEDDING_DIM_INPUT)
    print(f"  Embedded {len(embeddings_cache)} texts")

    # Step 3: Build graphs and convert to PyG
    print(f"\nStep 3: Building graphs and converting to PyG...")
    pyg_triplets = []

    for entry in tqdm(processed):
        try:
            triplet_pyg = {}
            for role in ["anchor", "similar", "dissimilar"]:
                ext = entry[role]["extraction"]
                from utils.extraction import NarrativeExtractionOutput
                extraction = NarrativeExtractionOutput(**ext)
                summary = entry[role]["summary"]

                G = create_narrative_graph(extraction, summary, embeddings_cache)
                pyg_data = nx_to_pyg_hetero(G)

                key = role + "_data"
                triplet_pyg[key] = pyg_data

            triplet_pyg["triplet_id"] = entry["triplet_id"]
            pyg_triplets.append(triplet_pyg)
        except Exception as e:
            print(f"  Error building graph for triplet {entry['triplet_id']}: {e}")
            continue

    # Save PyG data
    pyg_path = config.PREPROCESSED_DIR / "pyg_data.pkl"
    with open(pyg_path, "wb") as f:
        pickle.dump(pyg_triplets, f)
    print(f"\nSaved {len(pyg_triplets)} PyG triplets to {pyg_path}")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
