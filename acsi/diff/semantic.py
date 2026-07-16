from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from acsi.diff.deterministic import DiffResponse, deterministic_pair_equivalence

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FAKE_EMBEDDING_DIMS = 256
FAKE_FEATURES_PER_TOKEN = 4
TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FakeEmbedder:
    def __init__(self, dims: int = FAKE_EMBEDDING_DIMS) -> None:
        self.dims = dims

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in tokenize(text):
                for projection in range(FAKE_FEATURES_PER_TOKEN):
                    digest = hashlib.sha256(f"{token}:{projection}".encode()).digest()
                    column = int.from_bytes(digest[:4], "big") % self.dims
                    sign = 1.0 if digest[4] % 2 == 0 else -1.0
                    matrix[row_index, column] += sign / FAKE_FEATURES_PER_TOKEN**0.5
        return l2_normalize(matrix)


class RealEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(embeddings, dtype=np.float32))


@dataclass(frozen=True)
class PairClassification:
    deterministic_equal: bool
    similarity: float
    beyond_noise: bool
    reason: str | None = None


def classify_pair(
    baseline: DiffResponse,
    candidate: DiffResponse,
    *,
    embedder: EmbeddingClient,
    threshold: float,
) -> PairClassification:
    deterministic = deterministic_pair_equivalence(baseline, candidate)
    if deterministic.equivalent:
        return PairClassification(
            deterministic_equal=True,
            similarity=1.0,
            beyond_noise=False,
            reason=deterministic.reason,
        )
    embeddings = embedder.embed([baseline.text or "", candidate.text or ""])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return PairClassification(
        deterministic_equal=False,
        similarity=similarity,
        beyond_noise=similarity < threshold,
        reason=None,
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    value = float(np.dot(left.astype(np.float32), right.astype(np.float32)))
    return max(-1.0, min(1.0, value))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return (matrix / safe_norms).astype(np.float32)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
