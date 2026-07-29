from __future__ import annotations

import math

import pytest

from modules.config import load_config
from modules.contracts import Gripper, Query
from modules.hardware import fabricate_records
from modules.prediction import _generation_payload
from modules.retrieval import (
    ExperienceIndex,
    RetrievalMode,
    build_embedding_text,
    normalized_weights,
    s_contact,
    s_mass,
    s_roughness,
)
from tests.fakes import FakeEmbeddingProvider

CFG = load_config()


def test_similarity_bounds():
    assert s_mass(100, 100, CFG.retrieval.sigma_mass) == 1.0
    assert 0 <= s_mass(100, 900, CFG.retrieval.sigma_mass) <= 1
    assert s_contact(0.5, 0.5, CFG.retrieval.sigma_contact) == 1.0
    scale = CFG.roughness.characteristic_scale
    assert s_roughness(347.82, 347.82, CFG) == 1.0
    assert s_roughness(0.0, scale, CFG) == pytest.approx(math.exp(-1.0))
    assert 0 < s_roughness(0.0, 4 * scale, CFG) < s_roughness(0.0, scale, CFG)


def test_embedding_text_is_semantic_only():
    first = build_embedding_text("smooth glass cup", 100, 1, 0.9, CFG)
    second = build_embedding_text("smooth glass cup", 900, 5, 0.2, CFG)
    assert first == second == "smooth glass cup"


def test_weights_normalize_and_breakdown_sums_to_score():
    cfg = load_config().model_copy(deep=True)
    weights = normalized_weights(cfg)
    assert sum(weights.values()) == 1.0
    records = fabricate_records(cfg, 10)
    index = ExperienceIndex(cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)).fit(records)
    probe = records[0]
    q = Query(object_id="probe", image_path="", mass_g=probe.mass_g,
              roughness_index=probe.roughness_index,
              projected_contact_fraction=probe.projected_contact_fraction,
              semantic_description=probe.semantic_description)
    result = index.retrieve_objects(q, index.embed_query(q))[0]
    contributions = (
        result.similarity.semantic_contribution
        + result.similarity.mass_contribution
        + result.similarity.roughness_contribution
        + result.similarity.contact_contribution
    )
    assert abs(result.score - contributions) < 1e-9
    assert result.rank == 1


def test_paired_payload_can_omit_contact():
    cfg = load_config()
    records = fabricate_records(cfg, 10)
    index = ExperienceIndex(cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)).fit(records)
    q = Query(object_id="probe", image_path="", mass_g=200, roughness_index=2,
              projected_contact_fraction=0.7, semantic_description="x")
    qv = index.provider.embed("x")
    result = index.retrieve_objects(q, qv)[0]
    assert "projected_contact_fraction" in result.to_payload(include_contact=True)
    assert "projected_contact_fraction" not in result.to_payload(include_contact=False)


def test_object_retrieval_ranks_once_and_carries_both_gripper_labels():
    cfg = load_config().model_copy(deep=True)
    cfg.retrieval.k = 5
    records = fabricate_records(cfg, 20)
    index = ExperienceIndex(cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)).fit(records)
    probe = records[0]
    query = Query(
        object_id=probe.object_id,
        image_path="",
        mass_g=probe.mass_g,
        roughness_index=probe.roughness_index,
        projected_contact_fraction=probe.projected_contact_fraction,
        semantic_description=probe.semantic_description,
    )

    results = index.retrieve_objects(
        query,
        index.embed_query(query),
        exclude_object_id=probe.object_id,
    )

    assert len(results) == 5
    assert len({item.object_id for item in results}) == 5
    assert all(item.object_id != probe.object_id for item in results)
    assert all(item.gecko_min_force_n is not None for item in results)
    assert all(item.silicone_min_force_n is not None for item in results)
    assert results == sorted(results, key=lambda item: (-item.score, item.object_id))
    assert [item.rank for item in results] == [1, 2, 3, 4, 5]


def test_surface_retrieval_embeds_once_and_scores_sibling_conditions_separately():
    cfg = load_config().model_copy(deep=True)
    cfg.retrieval.k = 2
    cfg.retrieval.conditions_per_surface = 3
    baseline = fabricate_records(cfg, 2)
    first_id = baseline[0].object_id
    first = [record for record in baseline if record.object_id == first_id]
    siblings = []
    for number, mass in ((2, 400.0), (3, 900.0), (4, 1500.0)):
        siblings.extend(
            record.model_copy(
                update={
                    "object_id": f"{first_id}__condition_{number}",
                    "surface_id": first_id,
                    "condition_id": f"condition_{number}",
                    "mass_g": mass,
                    "roughness_index": 100.0 * number,
                    "projected_contact_fraction": 0.1 * number,
                }
            )
            for record in first
        )
    provider = FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    index = ExperienceIndex(cfg, provider).fit([*baseline, *siblings])
    assert provider.client.embedding_calls == 2

    query = Query(
        object_id="query",
        surface_id="query",
        image_path="",
        mass_g=900,
        roughness_index=300,
        projected_contact_fraction=0.3,
        semantic_description=first[0].semantic_description,
    )
    results = index.retrieve_objects(query, index.embed_query(query))
    first_surface = [item for item in results if item.surface_id == first_id]
    assert len(first_surface) == 3
    assert first_surface[0].condition_id == "condition_3"
    assert len({item.score for item in first_surface}) == 3
    assert {item.surface_rank for item in results} == {1, 2}


def test_e3_grouped_payload_hides_condition_identity_and_physical_fields():
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 3)
    first_id = records[0].object_id
    first = [record for record in records if record.object_id == first_id]
    records.extend(
        record.model_copy(
            update={
                "object_id": f"{first_id}__condition_2",
                "surface_id": first_id,
                "condition_id": "condition_2",
                "mass_g": 999,
            }
        )
        for record in first
    )
    index = ExperienceIndex(
        cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    ).fit(records)
    query = Query(
        object_id="query",
        image_path="",
        semantic_description=first[0].semantic_description,
    )
    retrieved = index.retrieve_objects(
        query, index.embed_query(query), mode=RetrievalMode.SEMANTIC_ONLY
    )
    payload = _generation_payload(
        cfg,
        query,
        retrieved,
        active_grippers=(Gripper.GECKO,),
        include_measured=False,
        include_retrieval=True,
        retrieval_mode=RetrievalMode.SEMANTIC_ONLY,
    )
    serialized = str(payload["retrieved_objects"])
    for forbidden in (
        "condition_id",
        "object_id",
        "mass_g",
        "roughness_index",
        "projected_contact_fraction",
    ):
        assert forbidden not in serialized
