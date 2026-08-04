from __future__ import annotations

import math

import pytest

from modules.config import load_config
from modules.contracts import Gripper, Query
from modules.datasets.paired_csv import load_rows, to_experiences
from modules.hardware import fabricate_records
from modules.prediction import _generation_payload
from modules.retrieval import (
    ExperienceIndex,
    RetrievalMode,
    build_embedding_text,
    normalized_weights,
    s_mass,
    s_roughness,
)
from tests.fakes import FakeEmbeddingProvider

CFG = load_config()


def test_similarity_bounds():
    assert s_mass(100, 100, CFG.retrieval.sigma_mass) == 1.0
    assert 0 <= s_mass(100, 900, CFG.retrieval.sigma_mass) <= 1
    scale = CFG.roughness.characteristic_scale
    assert s_roughness(347.82, 347.82, CFG) == 1.0
    assert s_roughness(0.0, scale, CFG) == pytest.approx(math.exp(-1.0))
    assert 0 < s_roughness(0.0, 4 * scale, CFG) < s_roughness(0.0, scale, CFG)


def test_embedding_text_is_semantic_only():
    assert build_embedding_text("smooth glass cup") == "smooth glass cup"


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


def test_visibility_aware_conditions_exclude_hidden_and_mixed_changes():
    cfg = load_config().model_copy(deep=True)
    cfg.retrieval.k = 20
    cfg.retrieval.conditions_per_surface = 3
    records = fabricate_records(cfg, 4)
    surface_id = records[0].object_id
    baseline_records = [record for record in records if record.object_id == surface_id]
    baseline = baseline_records[0]

    def sibling(condition_id: str, **updates):
        return [
            record.model_copy(
                update={
                    "object_id": f"{surface_id}__{condition_id}",
                    "surface_id": surface_id,
                    "condition_id": condition_id,
                    **updates,
                }
            )
            for record in baseline_records
        ]

    records.extend(sibling("condition_2", mass_g=baseline.mass_g + 100.0))
    records.extend(
        sibling("condition_3", roughness_index=baseline.roughness_index + 100.0)
    )
    records.extend(
        sibling(
            "condition_4",
            projected_contact_fraction=max(
                0.0, baseline.projected_contact_fraction - 0.1
            ),
        )
    )
    records.extend(
        sibling(
            "condition_5",
            mass_g=baseline.mass_g + 50.0,
            projected_contact_fraction=max(
                0.0, baseline.projected_contact_fraction - 0.2
            ),
        )
    )
    records.extend(sibling("condition_6"))

    index = ExperienceIndex(
        cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    ).fit(records)
    query = Query(
        object_id="query",
        image_path="",
        mass_g=baseline.mass_g + 50.0,
        roughness_index=baseline.roughness_index,
        projected_contact_fraction=baseline.projected_contact_fraction,
        semantic_description=baseline.semantic_description,
    )
    query_vec = index.embed_query(query)

    e4 = index.retrieve_objects(
        query,
        query_vec,
        mode=RetrievalMode.HYBRID,
        ranking_features=("semantic", "mass"),
        visible_condition_fields=("mass_g",),
    )
    e5 = index.retrieve_objects(
        query,
        query_vec,
        mode=RetrievalMode.HYBRID,
        ranking_features=("semantic", "mass", "roughness"),
        visible_condition_fields=("mass_g", "roughness_index"),
    )

    assert {
        item.condition_id for item in e4 if item.surface_id == surface_id
    } == {"baseline", "condition_2"}
    assert {
        item.condition_id for item in e5 if item.surface_id == surface_id
    } == {"baseline", "condition_2", "condition_3"}
    assert all(
        item.condition_id not in {"condition_4", "condition_5", "condition_6"}
        for item in e5
        if item.surface_id == surface_id
    )
    displayed_e5 = [item for item in e5 if item.surface_id == surface_id]
    assert displayed_e5[0].surface_score is not None
    assert displayed_e5[0].surface_score > max(item.score for item in displayed_e5)


def test_e6_uses_e5_surface_ranking_and_exposes_contact_variants():
    cfg = load_config().model_copy(deep=True)
    cfg.retrieval.k = 5
    cfg.retrieval.conditions_per_surface = 3
    records = fabricate_records(cfg, 8)
    surface_id = records[0].object_id
    baseline_records = [record for record in records if record.object_id == surface_id]
    baseline = baseline_records[0]
    for number, contact in ((2, 0.5), (3, 0.1)):
        records.extend(
            record.model_copy(
                update={
                    "object_id": f"{surface_id}__condition_{number}",
                    "surface_id": surface_id,
                    "condition_id": f"condition_{number}",
                    "projected_contact_fraction": contact,
                }
            )
            for record in baseline_records
        )

    index = ExperienceIndex(
        cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    ).fit(records)
    query = Query(
        object_id="query",
        image_path="",
        mass_g=baseline.mass_g,
        roughness_index=baseline.roughness_index,
        projected_contact_fraction=1.0,
        semantic_description=baseline.semantic_description,
    )
    query_vec = index.embed_query(query)
    ranking_features = ("semantic", "mass", "roughness")
    e5 = index.retrieve_objects(
        query,
        query_vec,
        ranking_features=ranking_features,
        visible_condition_fields=("mass_g", "roughness_index"),
    )

    e6 = index.retrieve_objects(
        query,
        query_vec,
        ranking_features=ranking_features,
        visible_condition_fields=(
            "mass_g",
            "roughness_index",
            "projected_contact_fraction",
        ),
    )

    def ranked_surfaces(items):
        return [
            item.surface_id
            for item in items
            if item.condition_rank == 1
        ]

    assert ranked_surfaces(e5) == ranked_surfaces(e6)
    surface_conditions = [item for item in e6 if item.surface_id == surface_id]
    assert [item.condition_id for item in surface_conditions] == [
        "baseline",
        "condition_2",
        "condition_3",
    ]
    expected_deltas = (
        0.5 - baseline.projected_contact_fraction,
        0.1 - baseline.projected_contact_fraction,
    )
    for item, expected_delta in zip(
        surface_conditions[1:], expected_deltas, strict=True
    ):
        comparison = item.comparison_to_baseline
        assert item.condition_role == "controlled_variant"
        assert comparison is not None
        assert comparison.changed_fields == ("projected_contact_fraction",)
        assert comparison.deltas["projected_contact_fraction"] == pytest.approx(
            expected_delta
        )
        # The variants only change contact, so the measured gecko force is
        # unchanged and its within-surface delta is exactly zero.
        assert comparison.force_deltas["gecko_min_force_delta_n"] == pytest.approx(
            0.0
        )

    payload = _generation_payload(
        cfg,
        query,
        e6,
        active_grippers=(Gripper.GECKO,),
        include_measured=True,
        include_retrieval=True,
        retrieval_mode=RetrievalMode.HYBRID,
        ranking_features=ranking_features,
        visible_condition_fields=(
            "mass_g",
            "roughness_index",
            "projected_contact_fraction",
        ),
    )
    retrieval_config = payload["retrieval_config"]
    assert set(retrieval_config["normalized_weights"]) == {
        "semantic",
        "mass",
        "roughness",
    }
    assert retrieval_config["projected_contact_used_for_surface_ranking"] is False
    surface_payload = next(
        item
        for item in payload["retrieved_objects"]
        if item["surface_id"] == surface_id
    )
    assert surface_payload["schema_version"] == 3
    assert [item["condition_role"] for item in surface_payload["conditions"]] == [
        "baseline",
        "controlled_variant",
        "controlled_variant",
    ]
    assert surface_payload["conditions"][1]["comparison_to_baseline"][
        "changed_fields"
    ] == ["projected_contact_fraction"]
    assert (
        "gecko_min_force_delta_n"
        in surface_payload["conditions"][1]["comparison_to_baseline"]["force_deltas"]
    )
    # A neighbor baseline's absolute contact is the cross-object confound E6 excludes;
    # it is suppressed while the within-surface variant keeps its contact evidence.
    assert "projected_contact_fraction" not in surface_payload["conditions"][0]
    assert "projected_contact_fraction" in surface_payload["conditions"][1]


def test_matforce_contact_sweeps_are_e6_evidence_but_not_e5_neighbors():
    cfg = load_config().model_copy(deep=True)
    cfg.dataset_id = "MatForceFinal"
    cfg.retrieval.k = 100
    rows = load_rows(cfg)
    records = to_experiences(cfg, [row for row in rows if row.split == "train"])
    index = ExperienceIndex(
        cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    ).fit(records)
    source = next(
        row
        for row in rows
        if row.object_name == "Large Cardboard Box"
        and row.condition_id == "baseline"
    )
    query = Query(
        object_id="validation_query",
        image_path="",
        mass_g=source.mass_g,
        roughness_index=source.roughness_index,
        projected_contact_fraction=source.projected_contact_fraction,
        semantic_description=source.object_name,
    )
    query_vec = index.embed_query(query)
    ranking_features = ("semantic", "mass", "roughness")
    e5 = index.retrieve_objects(
        query,
        query_vec,
        ranking_features=ranking_features,
        visible_condition_fields=("mass_g", "roughness_index"),
    )
    e6 = index.retrieve_objects(
        query,
        query_vec,
        ranking_features=ranking_features,
        visible_condition_fields=(
            "mass_g",
            "roughness_index",
            "projected_contact_fraction",
        ),
    )

    def ranked_surfaces(items):
        return [item.surface_id for item in items if item.condition_rank == 1]

    assert ranked_surfaces(e5) == ranked_surfaces(e6)
    for surface_id, expected_delta in {
        "large_cardboard_box": -0.5,
        "creatine": -0.9,
        "beaker": -0.6,
    }.items():
        assert [
            item.condition_id for item in e5 if item.surface_id == surface_id
        ] == ["baseline"]
        e6_conditions = [item for item in e6 if item.surface_id == surface_id]
        assert [item.condition_id for item in e6_conditions] == [
            "baseline",
            "condition_2",
        ]
        comparison = e6_conditions[1].comparison_to_baseline
        assert comparison is not None
        assert comparison.changed_fields == ("projected_contact_fraction",)
        assert comparison.deltas["projected_contact_fraction"] == pytest.approx(
            expected_delta
        )


def test_multiple_conditions_require_a_baseline():
    cfg = load_config().model_copy(deep=True)
    records = fabricate_records(cfg, 2)
    surface_id = records[0].object_id
    records = [
        record.model_copy(
            update={
                "condition_id": "condition_2",
                "object_id": f"{surface_id}__condition_2",
                "surface_id": surface_id,
            }
        )
        for record in records
        if record.object_id == surface_id
    ]
    first_condition = list(records)
    records.extend(
        record.model_copy(
            update={
                "condition_id": "condition_3",
                "object_id": f"{surface_id}__condition_3",
                "surface_id": surface_id,
            }
        )
        for record in first_condition
    )

    index = ExperienceIndex(
        cfg, FakeEmbeddingProvider(cfg.retrieval.embedding.dim)
    ).fit(records)
    query = Query(
        object_id="query",
        image_path="",
        mass_g=records[0].mass_g,
        semantic_description=records[0].semantic_description,
    )
    with pytest.raises(ValueError, match="multiple conditions but no baseline"):
        index.retrieve_objects(
            query,
            index.embed_query(query),
            ranking_features=("semantic", "mass"),
            visible_condition_fields=("mass_g",),
        )


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
