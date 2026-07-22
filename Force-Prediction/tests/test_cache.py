from __future__ import annotations

import numpy as np

from force_prediction.config import load_config
from force_prediction.llm import GeminiClient


class CountingClient(GeminiClient):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embedding_calls = 0

    def _embed_live(self, text, img_b64):
        self.embedding_calls += 1
        return np.ones(self.cfg.retrieval.embedding.dim, dtype=np.float32)


def test_embedding_cache_reuses_exact_request_and_invalidates_changes(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.paths.cache = "cache"
    cfg.models.cache = True
    client = CountingClient(cfg)

    first = client.embed(text="smooth glass")
    second = client.embed(text="smooth glass")
    client.embed(text="rough paper")

    assert np.array_equal(first, second)
    assert client.embedding_calls == 2
    assert client.cache_stats()["hits"] == 1
    assert client.cache_stats()["writes"] == 2

