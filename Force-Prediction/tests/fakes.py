"""Explicit test doubles for Gemini-backed production interfaces."""

from __future__ import annotations

import hashlib

import numpy as np

from modules.contracts import Gripper, JointGripperPrediction, PerGripperPrediction
from modules.perception import Description


class FakeGeminiClient:
    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim
        self.generation_calls = 0
        self.embedding_calls = 0

    def generate_json(self, *, schema, **_kwargs):  # noqa: ANN001, ANN003
        self.generation_calls += 1
        if schema is Description:
            return Description(
                retrieval_description="test Gemini object descriptor",
                contact_region="lateral grasp band",
                contact_material="test material",
            ).model_dump(mode="json")
        if schema is JointGripperPrediction:
            return JointGripperPrediction(
                gecko=PerGripperPrediction(
                    candidate_gripper=Gripper.GECKO,
                    predicted_normal_force_n=1.0,
                    reasoning_trace="test Gemini response",
                ),
                silicone=PerGripperPrediction(
                    candidate_gripper=Gripper.SILICONE,
                    predicted_normal_force_n=1.2,
                    reasoning_trace="test Gemini response",
                ),
                recommended_gripper="gecko",
                recommendation_summary="test Gemini recommendation",
            ).model_dump(mode="json")
        if schema is PerGripperPrediction:
            active = _kwargs.get("extra", {}).get("active_grippers", ["gecko"])
            return PerGripperPrediction(
                candidate_gripper=Gripper(active[0]),
                predicted_normal_force_n=1.0,
                reasoning_trace="test Gemini response",
            ).model_dump(mode="json")
        raise AssertionError(f"unsupported Gemini schema in test: {schema}")

    def embed(self, *, text: str, image_bgr=None):  # noqa: ANN001, ANN201
        del image_bgr
        self.embedding_calls += 1
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        vector = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-12)

    def cache_stats(self) -> dict:
        return {
            "backend_attempts": {
                "generation": self.generation_calls,
                "embedding": self.embedding_calls,
            }
        }


class FakeEmbeddingProvider:
    def __init__(self, dim: int = 1536) -> None:
        self.client = FakeGeminiClient(dim)

    def embed(self, text: str, image_bgr=None, is_query: bool = False):  # noqa: ANN001, ANN201
        prefix = "query" if is_query else "document"
        return self.client.embed(text=f"{prefix}: {text}", image_bgr=image_bgr)


def install_gemini_fakes(monkeypatch, dim: int = 1536) -> FakeGeminiClient:  # noqa: ANN001
    client = FakeGeminiClient(dim)
    for target in (
        "modules.perception.get_client",
        "modules.prediction.get_client",
        "modules.retrieval.get_client",
        "modules.experiments.helper.get_client",
    ):
        monkeypatch.setattr(target, lambda _cfg, client=client: client)
    return client
