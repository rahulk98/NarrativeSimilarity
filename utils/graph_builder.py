"""Heterogeneous narrative graph construction from extracted components."""

import networkx as nx
import numpy as np
from typing import Dict, List

from .extraction import NarrativeExtractionOutput, normalize_embeddings
from .ollama_client import gen_ollama_embeddings


def batch_embed_texts(texts: List[str], model: str, dimensions: int = 384) -> Dict[str, np.ndarray]:
    """Batch embed all unique texts at once and return a text->embedding cache."""
    unique_texts = list(set(texts))
    if not unique_texts:
        return {}

    embeddings = gen_ollama_embeddings(unique_texts, model_name=model, dimensions=dimensions)
    cache = {}
    for text, emb in zip(unique_texts, embeddings):
        arr = np.array(emb, dtype=np.float32)
        norm = np.linalg.norm(arr)
        cache[text] = arr / norm if norm > 0 else arr
    return cache


def create_narrative_graph(
    extraction: NarrativeExtractionOutput,
    story_summary: str,
    embeddings_cache: Dict[str, np.ndarray],
) -> nx.DiGraph:
    """
    Build a heterogeneous narrative graph from extracted components.

    Node types: Story, Theme, Action, Outcome
    Edge types:
        theme_supports_story      (Theme -> Story)
        action_starts_story       (Action[0] -> Story)
        action_ends_story         (Action[-1] -> Story)
        next_action               (Action[i] -> Action[i+1])
        prev_action               (Action[i+1] -> Action[i])
        action_leads_to_outcome   (Action[-1] -> Outcome)
        outcome_reflects_story    (Outcome -> Story)
    """
    G = nx.DiGraph()

    # Story node (initialized with summary embedding)
    story_id = "Story_0"
    story_emb = embeddings_cache.get(story_summary, np.zeros(384, dtype=np.float32))
    G.add_node(story_id, type="Story", text=story_summary,
               embedding=story_emb, plot_type=extraction.plot_type)

    # Theme nodes -> Story
    for i, theme in enumerate(extraction.abstract_theme):
        tid = f"Theme_{i}"
        emb = embeddings_cache.get(theme, np.zeros(384, dtype=np.float32))
        G.add_node(tid, type="Theme", text=theme, embedding=emb)
        G.add_edge(tid, story_id, rel="theme_supports_story")

    # Action nodes with temporal edges
    action_ids = []
    for i, action in enumerate(extraction.course_of_action):
        aid = f"Action_{i}"
        emb = embeddings_cache.get(action, np.zeros(384, dtype=np.float32))
        G.add_node(aid, type="Action", text=action, embedding=emb, position=i)
        action_ids.append(aid)

        if i == 0:
            G.add_edge(aid, story_id, rel="action_starts_story")
        if i > 0:
            G.add_edge(action_ids[i - 1], aid, rel="next_action")
            G.add_edge(aid, action_ids[i - 1], rel="prev_action")

    if action_ids:
        G.add_edge(action_ids[-1], story_id, rel="action_ends_story")

    # Outcome nodes -> Story
    for i, outcome in enumerate(extraction.outcome):
        oid = f"Outcome_{i}"
        emb = embeddings_cache.get(outcome, np.zeros(384, dtype=np.float32))
        G.add_node(oid, type="Outcome", text=outcome, embedding=emb)
        G.add_edge(oid, story_id, rel="outcome_reflects_story")
        if action_ids:
            G.add_edge(action_ids[-1], oid, rel="action_leads_to_outcome")

    return G
