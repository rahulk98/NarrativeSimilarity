"""Narrative component extraction via LLM with Pydantic validation."""

from typing import List
from pydantic import BaseModel
import numpy as np

from .ollama_client import gen_ollama, gen_ollama_embeddings


class NarrativeExtractionOutput(BaseModel):
    abstract_theme: List[str]
    course_of_action: List[str]
    outcome: List[str]
    plot_type: str


class StorySummaryOutput(BaseModel):
    summary: str


EXTRACTION_PROMPT = """Extract narrative components from this story:

Story: {story}

Extract:
1. Abstract themes - multiple high-level conceptual phrases (3-8 words each) covering different conceptual dimensions (emotional, situational, existential, relational, moral), ordered by importance
2. Course of action - chronological sequence of major actions, turning points, and causal developments (short, factual, action-focused steps)
3. Outcome - final state of the narrative: how conflicts conclude, final fates, morals/lessons; exclude intermediate states
4. Plot type (Tragedy, Comedy, Quest, Rebirth, Overcoming the Monster, Rags to Riches, Other)

Return as JSON with keys: abstract_theme, course_of_action, outcome, plot_type"""


def extract_narrative_components(story: str, model: str) -> NarrativeExtractionOutput:
    """Extract themes, actions, outcomes, and plot type from a story."""
    response = gen_ollama(
        prompt=EXTRACTION_PROMPT.format(story=story),
        temperature=0.0,
        model_name=model,
        system_instruction="You are an expert at extracting structured narrative information from stories. Always output valid JSON.",
        json_schema=NarrativeExtractionOutput.model_json_schema(),
    )
    if not response:
        raise ValueError("Ollama returned empty response")
    return NarrativeExtractionOutput.model_validate_json(response)


def generate_story_summary(story: str, model: str, max_words: int = 15) -> str:
    """Generate a concise story summary for narrative node initialization."""
    prompt = f"Summarize this story in exactly {max_words} words or less. Be concise and capture the core narrative.\n\nStory: {story}\n\nSummary:"
    response = gen_ollama(
        prompt=prompt,
        temperature=0.0,
        model_name=model,
        system_instruction="You are an expert at creating concise story summaries. Output only the summary text.",
        json_schema=StorySummaryOutput.model_json_schema(),
    )
    if not response:
        return " ".join(story.split()[:max_words]) + "..."
    return StorySummaryOutput.model_validate_json(response).summary


def generate_embeddings(texts: List[str], model: str, dimensions: int = 384) -> np.ndarray:
    """Generate embeddings for a list of texts via Ollama."""
    embeddings = gen_ollama_embeddings(texts, model_name=model, dimensions=dimensions)
    return np.array(embeddings, dtype=np.float32)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings to unit length."""
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return embeddings / norms
