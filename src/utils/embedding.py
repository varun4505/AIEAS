"""Sentence embedding utilities."""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the SentenceTransformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise RuntimeError("sentence-transformers is required for embeddings") from exc

    return SentenceTransformer(MODEL_NAME)


def compute_embeddings(texts: str | Iterable[str]) -> np.ndarray:
    """Compute normalized sentence embeddings for one or many texts."""
    model = _load_model()
    if isinstance(texts, str):
        sentences = [texts]
    else:
        sentences = list(texts)

    embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings
