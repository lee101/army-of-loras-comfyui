"""pybed static int8 embeddings, loaded once and cached per phrase.

pybed (https://github.com/lee101/pybed, MIT) is a token-lookup embedder: no transformer
forward pass, ~0.01 ms per phrase on CPU, so every routing decision can afford to score
hundreds of LoRA descriptions. The 16 MB model ships in ../embed_model.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "embed_model"

_lock = threading.Lock()
_model = None


def model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from pybed import EmbedModel

                _model = EmbedModel.from_dir(MODEL_DIR)
    return _model


@lru_cache(maxsize=8192)
def embed(text: str) -> np.ndarray:
    """Unit-normalised float32 embedding of `text` (zeros for empty text)."""
    v = model().embed(text.lower()).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def embed_many(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 512), dtype=np.float32)
    return np.stack([embed(t) for t in texts])


def best_cos(query: np.ndarray, texts: list[str]) -> tuple[float, str]:
    """Highest cosine between `query` and any of `texts`, with the phrase that won."""
    if not texts:
        return 0.0, ""
    sims = embed_many(texts) @ query
    i = int(np.argmax(sims))
    return float(sims[i]), texts[i]
