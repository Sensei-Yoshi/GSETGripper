from __future__ import annotations

import numpy as np

from force_prediction.config import load_config
from force_prediction.contracts import Gripper, JointGripperPrediction, PerGripperPrediction
from force_prediction.llm import GeminiClient


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
