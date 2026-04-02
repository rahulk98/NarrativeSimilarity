"""Ollama API wrapper for text generation and embedding."""

from typing import Dict, Any, Optional
from ollama import chat, embed


def gen_ollama(
    prompt: str,
    temperature: float = 0.0,
    model_name: str = "gpt-oss:20b",
    json_schema: Optional[Dict[str, Any]] = None,
    system_instruction: Optional[str] = None,
) -> str | None:
    """Generate text using Ollama chat API with optional JSON schema."""
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature}

    try:
        kwargs = dict(model=model_name, messages=messages, options=options)
        if json_schema:
            kwargs["format"] = json_schema

        response = chat(**kwargs)

        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content
        return None
    except Exception as e:
        print(f"Ollama generation error: {e}")
        return None


def gen_ollama_embeddings(
    texts: list[str],
    model_name: str = "all-minilm:33m",
    dimensions: int = 384,
) -> list[list[float]]:
    """Generate embeddings for a list of texts using Ollama."""
    try:
        embeddings = []
        for text in texts:
            response = embed(model=model_name, input=text, dimensions=dimensions)
            if hasattr(response, "embeddings"):
                embeddings.append(response.embeddings[0])
            else:
                return []
        return embeddings
    except Exception as e:
        print(f"Ollama embedding error: {e}")
        return []
