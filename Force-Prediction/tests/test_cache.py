from __future__ import annotations

import numpy as np

from modules.config import load_config
from modules.contracts import Gripper, JointGripperPrediction, PerGripperPrediction
from modules.models.gemini import GeminiClient


class CountingClient(GeminiClient):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embedding_calls = 0

    def _embed_live(self, text, img_b64):
        self.embedding_calls += 1
        return np.ones(self.cfg.retrieval.embedding.dim, dtype=np.float32)


class CountingGenerationClient(GeminiClient):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.generation_calls = 0

    def _generate_json_live(self, system, instruction, schema, img_b64, extra):
        self.generation_calls += 1
        return JointGripperPrediction(
            gecko=PerGripperPrediction(
                candidate_gripper=Gripper.GECKO, predicted_normal_force_n=1.0
            ),
            silicone=PerGripperPrediction(
                candidate_gripper=Gripper.SILICONE, predicted_normal_force_n=1.2
            ),
            recommended_gripper="gecko",
        ).model_dump(mode="json")


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


def test_generation_cache_reuses_joint_request_and_invalidates_payload(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.paths.cache = "cache"
    cfg.models.cache = True
    client = CountingGenerationClient(cfg)
    kwargs = {
        "system": "system",
        "instruction": "predict both",
        "schema": JointGripperPrediction,
        "extra": {"retrieved_objects": [{"object_id": "a"}]},
    }

    first = client.generate_json(**kwargs)
    second = client.generate_json(**kwargs)
    client.generate_json(
        **{
            **kwargs,
            "extra": {"retrieved_objects": [{"object_id": "b"}]},
        }
    )

    assert first == second
    assert client.generation_calls == 2
    assert client.cache_stats()["hits"] == 1
    assert client.cache_stats()["writes"] == 2


def test_generation_bypass_keeps_embedding_cache_enabled(tmp_path):
    cfg = load_config().model_copy(deep=True)
    cfg.root = tmp_path
    cfg.paths.cache = "cache"
    cfg.models.cache = True
    cfg.models.bypass_generation_cache = True
    generation_client = CountingGenerationClient(cfg)
    generation_kwargs = {
        "system": "system",
        "instruction": "predict both",
        "schema": JointGripperPrediction,
        "extra": {"retrieved_objects": [{"object_id": "a"}]},
    }

    generation_client.generate_json(**generation_kwargs)
    generation_client.generate_json(**generation_kwargs)

    generation_stats = generation_client.cache_stats()
    assert generation_client.generation_calls == 2
    assert generation_stats["generation_bypassed"] is True
    assert generation_stats["generation"] == {
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "read_errors": 0,
        "legacy_hits": 0,
    }

    embedding_client = CountingClient(cfg)
    first = embedding_client.embed(text="smooth glass")
    second = embedding_client.embed(text="smooth glass")

    assert np.array_equal(first, second)
    assert embedding_client.embedding_calls == 1
    assert embedding_client.cache_stats()["embeddings"]["hits"] == 1
