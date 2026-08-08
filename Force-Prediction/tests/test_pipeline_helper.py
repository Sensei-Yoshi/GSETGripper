from __future__ import annotations

import numpy as np
import pytest

from modules.config import load_config
from modules.contracts import Gripper, PerGripperPrediction
from modules.pipeline import QueryInput, predict_gripper_force, query_input_from_object
from tests.factories import fabricate_records
from tests.fakes import FakeEmbeddingProvider, install_gemini_fakes


def _query(records, cfg):  # noqa: ANN001, ANN202
    query = query_input_from_object(records, cfg)
    query.image_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    return query


@pytest.mark.parametrize(
    ("experiment", "expected"),
    ((5, "e5"), ("e5", "e5")),
)
def test_predict_gripper_force_scopes_pipeline_to_one_gripper(
    experiment, expected, monkeypatch
):  # noqa: ANN001
    cfg = load_config().model_copy(deep=True)
    install_gemini_fakes(monkeypatch, cfg.retrieval.embedding.dim)
    records = fabricate_records(cfg, 12)
    held = records[0].object_id
    training = [record for record in records if record.object_id != held]
    query_rows = [record for record in records if record.object_id == held]

    result = predict_gripper_force(
        cfg,
        experiment,
        Gripper.SILICONE,
        training,
        _query(query_rows, cfg),
    )

    assert result.experiment_id == expected
    assert result.gripper is Gripper.SILICONE
    assert result.feasible
    assert result.force_n == result.prediction.predicted_normal_force_n
    assert result.prediction.candidate_gripper is Gripper.SILICONE
    assert cfg.prediction.active_grippers == (Gripper.GECKO, Gripper.SILICONE)


def test_predict_gripper_force_returns_none_for_infeasible_prediction(monkeypatch):
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 4)
    held = records[0].object_id

    class InfeasibleClient:
        def generate_json(self, **_kwargs):  # noqa: ANN003, ANN202
            return PerGripperPrediction(
                candidate_gripper=Gripper.GECKO,
                feasible=False,
                predicted_normal_force_n=9.0,
            ).model_dump(mode="json")

        def cache_stats(self) -> dict:
            return {}

    client = InfeasibleClient()
    monkeypatch.setattr("modules.prediction.get_client", lambda _cfg: client)
    monkeypatch.setattr("modules.experiments.helper.get_client", lambda _cfg: client)
    monkeypatch.setattr(
        "modules.experiments.helper.get_embedding_provider",
        lambda _cfg: FakeEmbeddingProvider(cfg.retrieval.embedding.dim),
    )

    result = predict_gripper_force(
        cfg,
        "e1",
        "gecko",
        [record for record in records if record.object_id != held],
        _query([record for record in records if record.object_id == held], cfg),
    )

    assert not result.feasible
    assert result.force_n is None


@pytest.mark.parametrize(
    ("experiment", "gripper"),
    ((6, "gecko"), (True, "gecko"), (1, "vacuum")),
)
def test_predict_gripper_force_rejects_invalid_inputs(experiment, gripper):  # noqa: ANN001
    cfg = load_config()
    with pytest.raises((KeyError, TypeError, ValueError)):
        predict_gripper_force(
            cfg,
            experiment,
            gripper,
            [],
            QueryInput(object_id="query", image_bgr=np.zeros((8, 8, 3), dtype=np.uint8)),
        )
