from __future__ import annotations

import os

import pytest

from acsi.diff.deterministic import DiffResponse
from acsi.diff.semantic import FakeEmbedder, RealEmbedder, classify_pair, cosine_similarity


def test_fake_embedder_similarity_properties() -> None:
    embedder = FakeEmbedder()
    vectors = embedder.embed(
        [
            "volunteer application summary coordinator",
            "volunteer application summary coordinator",
            "volunteer applicant summary coordinator",
            "broken json wrong schema failure",
            "zebra quantum marble",
        ]
    )

    assert cosine_similarity(vectors[0], vectors[1]) == 1.0
    assert cosine_similarity(vectors[0], vectors[4]) < 0.1
    assert cosine_similarity(vectors[0], vectors[2]) > cosine_similarity(vectors[0], vectors[3])


def test_classify_pair_skips_embeddings_for_deterministic_equal() -> None:
    classification = classify_pair(
        DiffResponse(text='{"a": 1}'),
        DiffResponse(text='{"a":1}'),
        embedder=FakeEmbedder(),
        threshold=0.9,
    )

    assert classification.deterministic_equal
    assert classification.similarity == 1.0
    assert not classification.beyond_noise


def test_classify_pair_uses_threshold_for_semantic_difference() -> None:
    classification = classify_pair(
        DiffResponse(text="volunteer application summary"),
        DiffResponse(text="zebra quantum marble"),
        embedder=FakeEmbedder(),
        threshold=0.9,
    )

    assert not classification.deterministic_equal
    assert classification.similarity < 0.1
    assert classification.beyond_noise


@pytest.mark.skipif(
    os.environ.get("ACSI_TEST_REAL_EMBEDDINGS") != "1",
    reason="set ACSI_TEST_REAL_EMBEDDINGS=1 to download and test the real embedder",
)
def test_real_embedder_integration() -> None:
    vectors = RealEmbedder().embed(["same text", "same text"])

    assert vectors.shape[0] == 2
    assert cosine_similarity(vectors[0], vectors[1]) > 0.99
